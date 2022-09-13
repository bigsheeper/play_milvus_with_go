package main

import (
	"context"
	"fmt"
	milvusClient "github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"sync"
	"time"
)

var (
	TopK   = []int{50}
	NQ     = []int{1}
	EF     = []int{50}
	NPROBE = []int{1}
	allQPS = 0.0
)

func Search(client milvusClient.Client, dataset, indexType string, process int, partitions []string) {
	var pList []int
	if indexType == "HNSW" {
		pList = EF
	} else if indexType == "IVF_FLAT" {
		pList = NPROBE
	} else {
		panic("illegal index type")
	}

	var dataPath string
	if dataset == "taip" || dataset == "zc" {
		dataPath = TaipDataPath
	} else if dataset == "sift" {
		dataPath = SiftDataPath
	} else {
		panic("wrong dataset")
	}
	for _, p := range pList {
		searchParams := newSearchParams(p, indexType)
		for _, nq := range NQ {
			vectors := generatedEntities(dataPath, nq)
			for _, topK := range TopK {
				allQPS = 0.0
				var wg sync.WaitGroup
				wg.Add(process)
				for g := 0; g < process; g++ {
					go func() {
						defer wg.Done()
						cost := int64(0)
						for i := 0; i < RunTime; i++ {
							start := time.Now()
							_, err := client.Search(context.Background(), dataset, partitions, "", []string{},
								vectors, VecFieldName, entity.L2, topK, searchParams)
							if err != nil {
								panic(err)
							}
							cost += time.Since(start).Microseconds()
						}
						avgTime := float64(cost/RunTime) / 1000.0 / 1000.0
						qps := float64(nq) / avgTime
						fmt.Printf("average search time: %f， vps: %f \n", avgTime, qps)
						allQPS += qps
					}()
				}
				wg.Wait()
				fmt.Printf("nq = %d, topK = %d, param = %d, goroutine = %d, vps = %f \n", nq, topK, p, process, allQPS)
			}
		}
	}
}

func newSearchParams(p int, indexType string) entity.SearchParam {
	if indexType == "HNSW" {
		searchParams, err := entity.NewIndexHNSWSearchParam(p)
		if err != nil {
			panic(err)
		}
		return searchParams
	} else if indexType == "IVF_FLAT" {
		searchParams, err := entity.NewIndexIvfFlatSearchParam(p)
		if err != nil {
			panic(err)
		}
		return searchParams
	}
	panic("illegal search params")
}
