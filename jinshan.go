package main

import (
	"context"
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

type Milvus struct {
	client *client.Client
}

func (c *Milvus) Upsert(ctx context.Context, param UpsertParam, columns ...entity.Column) error {
	if len(param.Ids) == 0 {
		return nil
	}
	//查询数据是否存在
	existId, err := c.QueryPksExist(ctx, QueryByPksParam{
		CollectionName: param.CollectionName,
		Ids:            param.Ids,
	})
	if err != nil {
		return errors.WithStack(err)
	}

	//删除存在的数据
	if len(existId) > 0 {
		err = c.Delete(ctx, param.CollectionName, existId...)
		if err != nil {
			return errors.WithStack(err)
		}
	}

	idColumn := entity.NewColumnInt64("id", param.Ids)

	columns = append([]entity.Column{idColumn}, columns...)
	_, err = c.client.Insert(ctx, param.CollectionName, "", columns...)
	return errors.WithStack(err)
}

func (c *Milvus) Search() {
	sr, err := c.client.Search(ctx,
		e.CollectionName,
		[]string{},
		e.Expr,
		[]string{"id"},
		[]entity.Vector{vector},
		"vectors",
		entity.L2, e.TopK, sp)
}
