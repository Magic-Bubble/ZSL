
# python train_gcn_relation.py --graph materials/imagenet-relation-graph-topk1.json --save-path log/kg/save/wup1 > log/kg/save/wup1/train.log
# python train_gcn_relation.py --graph materials/imagenet-relation-graph-topk3.json --save-path log/kg/save/wup3 > log/kg/save/wup3/train.log
# python train_gcn_relation.py --graph materials/imagenet-relation-graph-topk7.json --save-path log/kg/save/wup7 > log/kg/save/wup7/train.log
# python train_gcn_relation.py --graph materials/imagenet-relation-graph-topk9.json --save-path log/kg/save/wup9 > log/kg/save/wup9/train.log

# python evaluate_imagenet.py --pred log/kg/save/wup1/epoch-1500.pred --test-set 2-hops > log/kg/wup1-2-hops.txt
# python evaluate_imagenet.py --pred log/kg/save/wup1/epoch-1500.pred --test-set 3-hops > log/kg/wup1-3-hops.txt
# python evaluate_imagenet.py --pred log/kg/save/wup1/epoch-1500.pred --test-set all > log/kg/wup1-all.txt
# 
# python evaluate_imagenet.py --pred log/kg/save/wup3/epoch-1500.pred --test-set 2-hops > log/kg/wup3-2-hops.txt
# python evaluate_imagenet.py --pred log/kg/save/wup3/epoch-1500.pred --test-set 3-hops > log/kg/wup3-3-hops.txt
# python evaluate_imagenet.py --pred log/kg/save/wup3/epoch-1500.pred --test-set all > log/kg/wup3-all.txt
# 
python evaluate_imagenet.py --pred log/kg/save/wup5/epoch-1500.pred --test-set 2-hops > log/kg/wup5-2-hops.txt
python evaluate_imagenet.py --pred log/kg/save/wup5/epoch-1500.pred --test-set 3-hops > log/kg/wup5-3-hops.txt
python evaluate_imagenet.py --pred log/kg/save/wup5/epoch-1500.pred --test-set all > log/kg/wup5-all.txt

# python evaluate_imagenet.py --pred log/kg/save/wup7/epoch-1500.pred --test-set 2-hops > log/kg/wup7-2-hops.txt
# python evaluate_imagenet.py --pred log/kg/save/wup7/epoch-1500.pred --test-set 3-hops > log/kg/wup7-3-hops.txt
# python evaluate_imagenet.py --pred log/kg/save/wup7/epoch-1500.pred --test-set all > log/kg/wup7-all.txt
# 
# python evaluate_imagenet.py --pred log/kg/save/wup9/epoch-1500.pred --test-set 2-hops > log/kg/wup9-2-hops.txt
# python evaluate_imagenet.py --pred log/kg/save/wup9/epoch-1500.pred --test-set 3-hops > log/kg/wup9-3-hops.txt
# python evaluate_imagenet.py --pred log/kg/save/wup9/epoch-1500.pred --test-set all > log/kg/wup9-all.txt
