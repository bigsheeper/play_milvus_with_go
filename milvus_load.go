package main

import (
	"context"
	"fmt"
	milvusClient "github.com/xiaocai2333/milvus-sdk-go/v2/client"
	"time"
)

var (
	CurLoadRows   = 0
	TotalLoadRows = 0
)

func Load(client milvusClient.Client, dataset string) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go printLoadProgress(ctx)

	err := client.LoadCollection(ctx, dataset, false)
	if err != nil {
		print(err)
	}
	fmt.Printf("Load collection %s done!\n", dataset)

	//if len(partitions) == 0 {
	//	err := client.LoadCollection(ctx, dataset, false)
	//	if err != nil {
	//		print(err)
	//	}
	//	fmt.Printf("Load collection %s done!\n", dataset)
	//} else {
	//	err := client.LoadPartitions(ctx, dataset, partitions, false)
	//	if err != nil {
	//		panic(err)
	//	}
	//	fmt.Println("Load partitions done, partitions: ", partitions)
	//}

	confirmLoadComplete(client, dataset)
	return
}

func Release(client milvusClient.Client, dataset string) {
	if err := client.ReleaseCollection(context.Background(), dataset); err != nil {
		panic(err)
	}
	//if len(partitions) == 0 {
	//	if err := client.ReleaseCollection(context.Background(), dataset); err != nil {
	//		panic(err)
	//	}
	//} else {
	//	if err := client.ReleasePartitions(context.Background(), dataset, partitions); err != nil {
	//		panic(err)
	//	}
	//}
	return
}

func confirmLoadComplete(client milvusClient.Client, dataset string) {
	// TODO: sdk does not implement this function
}

func printLoadProgress(ctx context.Context) {
	ticker := time.NewTicker(time.Second)
	for {
		select {
		case <-ctx.Done():
			fmt.Println("\n Load done!")
			return
		case <-ticker.C:
			fmt.Print("loading...\r")
		}
	}
}
