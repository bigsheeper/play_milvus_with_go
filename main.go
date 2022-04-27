package main

import (
	"bufio"
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	milvusClient "github.com/xiaocai2333/milvus-sdk-go/v2/client"
	"github.com/xiaocai2333/milvus-sdk-go/v2/entity"
	"google.golang.org/grpc"
	"io"
	"math"
	"os"
	"path"
	"strconv"
	"strings"
	"time"
)

const (
	CollectionName       = "sift_0"
	DefaultPartitionName = "_default"
	VecFieldName         = "vec"

	TaipDataPath = "/data/milvus/raw_data/zjlab"
	SiftDataPath = "/home/sheep/data-mnt/milvus/raw_data/sift10m"
	QueryFile    = "query.npy"
)

var (
	Dim = 768
)

func createClient(addr string) milvusClient.Client {
	opts := []grpc.DialOption{grpc.WithInsecure(),
		grpc.WithBlock(),                   //block connect until healthy or timeout
		grpc.WithTimeout(20 * time.Second)} // set connect timeout to 2 Second
	client, err := milvusClient.NewGrpcClient(context.Background(), addr, opts...)
	if err != nil {
		panic(err)
	}
	return client
}

func BytesToFloat32(bits []byte) []float32 {
	vectors := make([]float32, 0)
	start, end := 0, 4
	for start < len(bits) && end <= len(bits) {
		num := math.Float32frombits(binary.LittleEndian.Uint32(bits[start:end]))
		vectors = append(vectors, num)
		start += 4
		end += 4
	}
	return vectors
}

func ReadBytesFromFile(nq int, filePath string) []byte {
	f, err := os.Open(filePath)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	r := bufio.NewReader(f)
	chunks := make([]byte, 0)
	buf := make([]byte, 4*Dim)
	readByte := 0
	for readByte < nq*Dim*4 {
		n, err := r.Read(buf)
		if err != nil && err != io.EOF {
			panic(err)
		}
		if 0 == n {
			break
		}
		chunks = append(chunks, buf[:n]...)
		readByte += n
	}
	//fmt.Println(len(chunks))
	return chunks[:nq*Dim*4]
}

func generatedEntities(dataPath string, nq int) []entity.Vector {
	filePath := path.Join(dataPath, QueryFile)
	bits := ReadBytesFromFile(nq, filePath)
	vectors := make([]entity.Vector, 0)
	for i := 0; i < nq; i++ {
		var vector entity.FloatVector = BytesToFloat32(bits[i*Dim*4 : (i+1)*Dim*4])
		//fmt.Println(len(vector))
		vectors = append(vectors, vector)
	}
	return vectors
}

func generateInsertFile(x int) string {
	return "binary_" + strconv.Itoa(Dim) + "d_" + fmt.Sprintf("%05d", x) + ".npy"
}

func generateInsertPath(dataPath string, x int) string {
	return path.Join(dataPath, generateInsertFile(x))
}

type Strings []string

func newSliceValue(vals []string, p *[]string) *Strings {
	*p = vals
	return (*Strings)(p)
}

func (s *Strings) Set(val string) error {
	*s = Strings(strings.Split(val, ","))
	return nil
}

func (s *Strings) Get() interface{} {
	return []string(*s)
}

func (s *Strings) String() string {
	return strings.Join([]string(*s), ",")
}

var (
	addr      string
	dataset   string
	indexType string
	process   int
	operation string
	//partitions []string

	argNQ                 int
	argSearchPartitionNum int
	argSearchRunTimes     int64

	globalPartitionNames []string
)

func init() {
	flag.StringVar(&addr, "host", "localhost:19530", "milvus addr")
	flag.StringVar(&dataset, "dataset", "sift", "dataset for test")
	flag.StringVar(&indexType, "index", "HNSW", "index type for collection, HNSW | IVF_FLAT | FLAT")
	flag.StringVar(&operation, "op", "", "what do you want to do")
	//flag.Var(newSliceValue([]string{}, &partitions), "p", "partitions which you want to load")
	flag.IntVar(&process, "process", 1, "goroutines for test")
	flag.IntVar(&argNQ, "nq", 1, "search nq")
	flag.IntVar(&argSearchPartitionNum, "search_partition_num", 10, "number of partitions to search")
	flag.Int64Var(&argSearchRunTimes, "run_times", 10, "times of running search")
}

func main() {
	flag.Parse()
	fmt.Printf("host: %s, dataset: %s, index: %s, op: %s\n", addr, dataset,
		indexType, operation)

	client := createClient(addr)
	defer client.Close()
	if dataset == "taip" || dataset == "zc" {
		Dim = 768
	} else if dataset == "sift" {
		Dim = 128
	}
	if operation == "Insert" {
		Insert(client, dataset, indexType)
	} else if operation == "Search" {
		fmt.Printf("process: %d, nq: %d, search_partition_num: %d, run_times: %d\n", process, argNQ,
			argSearchPartitionNum, argSearchRunTimes)
		Search(client, dataset, indexType, process, globalPartitionNames)
	} else if operation == "Index" {
		CreateIndex(client, dataset, indexType)
	} else if operation == "Load" {
		Load(client, dataset)
	} else if operation == "Release" {
		Release(client, dataset)
	} else if operation == "Show" {
		ShowInfos(client, dataset)
	} else {
		panic(fmt.Sprintf("Invalid op: %s, op should be one of ['Insert', 'Index', 'Load', 'Search', 'Release']", operation))
	}
	return
}

func ShowInfos(client milvusClient.Client, dataset string) {
	collectionName := dataset
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	fmt.Println("\nDescribeCollection...")
	desColRsp, err := client.DescribeCollection(ctx, collectionName)
	fmt.Println("collectionName:", desColRsp.Name)
	fmt.Println("collectionID:", desColRsp.ID)
	fmt.Println("collectionSchema:", desColRsp.Schema)
	fmt.Println("PhysicalChannels:", desColRsp.PhysicalChannels)
	fmt.Println("VirtualChannels:", desColRsp.VirtualChannels)

	fmt.Println("\nShowPartitions...")
	showParRsp, err := client.ShowPartitions(ctx, collectionName)
	if err != nil {
		panic(err)
	}
	for _, par := range showParRsp {
		fmt.Println(par)
	}

	getVectorField := func(schema *entity.Schema) *entity.Field {
		for _, field := range schema.Fields {
			if field.DataType == entity.FieldTypeFloatVector {
				return field
			}
		}
		panic("No vector field found!")
	}

	fmt.Println("\nDescribeIndex...")
	desIndexRsp, err := client.DescribeIndex(ctx, collectionName, getVectorField(desColRsp.Schema).Name)
	fmt.Println(desIndexRsp[0])
}
