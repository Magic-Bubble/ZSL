
for k in 1 3 7 9
do
python make_relation_graph.py --output imagenet-relation-graph-topk${k}.json --topk ${k}
done
