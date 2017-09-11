### pretreat-v1.py 

生成的texts.json对pdf进行区域划分，划分如下

```json
{
  texts: // 文本区域
  tables: // 表格区域
  {
  	lines: // 表格中的框线
  	cells: // 表格中的单元格区域
  	{
  	  texts: // 表格中的单元格中的文本
    }
  }
  curves: // 线条区域
  others: // 其他如图片、公式等区域
}
```



### pretreat-v2.py

生成的texts.json对pdf进行区域划分，划分如下

```json
{
  texts: // 文本区域
  tables: // 表格区域
  {
  	curves: // 表格中的框线
  	texts: // 表格中的单元格中的文本
  }
  curves: // 线条区域
  others: // 其他如图片、公式等区域
}
```