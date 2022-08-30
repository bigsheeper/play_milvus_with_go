package main

import (
	"context"
	milvusClient "github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

func CreateIndex(client milvusClient.Client, dataset string, indexType string) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if entity.IndexType(indexType) == entity.Flat {
		if err := client.DropIndex(ctx, dataset, VecFieldName); err != nil {
			panic(err)
		}
		return
	}
	_ = client.Flush(ctx, dataset, false)
	if dataset == "taip" || dataset == "sift" {
		if entity.IndexType(indexType) == entity.HNSW {
			if err := client.CreateIndex(ctx, dataset, VecFieldName, NewTaipHNSWIndex(), false); err != nil {
				panic(err)
			}
		} else if entity.IndexType(indexType) == entity.IvfFlat {
			if err := client.CreateIndex(ctx, dataset, VecFieldName, NewTaipIVFFLATIndex(), false); err != nil {
				panic(err)
			}
		}
	}
	return
}

func NewTaipHNSWIndex() *entity.IndexHNSW {
	indexParams, err := entity.NewIndexHNSW(entity.L2, 16, 256)
	if err != nil {
		panic(err)
	}
	return indexParams
}

func NewTaipIVFFLATIndex() *entity.IndexIvfFlat {
	indexParams, err := entity.NewIndexIvfFlat(entity.L2, 1024)
	if err != nil {
		panic(err)
	}
	return indexParams
}

func NewSiftHNSWIndex() *entity.IndexHNSW {
	indexParams, err := entity.NewIndexHNSW(entity.L2, 16, 256)
	if err != nil {
		panic(err)
	}
	return indexParams
}

func NewSiftIVFFLATIndex() *entity.IndexIvfFlat {
	indexParams, err := entity.NewIndexIvfFlat(entity.L2, 1024)
	if err != nil {
		panic(err)
	}
	return indexParams
}
