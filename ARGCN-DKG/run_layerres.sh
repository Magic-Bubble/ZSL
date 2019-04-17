
# python train_gcn_basic.py --hidden-layers "d" --no-residual --save-path log/layerres/save/layer1_nores_isa > log/layerres/save/layer1_nores_isa/train.log
# python train_gcn_basic.py --hidden-layers "d" --save-path log/layerres/save/layer1_res_isa > log/layerres/save/layer1_res_isa/train.log

# python train_gcn_basic.py --hidden-layers "d2048,d" --no-residual --save-path log/layerres/save/layer2_nores_isa > log/layerres/save/layer2_nores_isa/train.log
# python train_gcn_basic.py --hidden-layers "d2048,d" --save-path log/layerres/save/layer2_res_isa > log/layerres/save/layer2_res_isa/train.log
# 
# python train_gcn_basic.py --hidden-layers "d2048,d2048,d" --no-residual --save-path log/layerres/save/layer3_nores_isa > log/layerres/save/layer3_nores_isa/train.log
# python train_gcn_basic.py --hidden-layers "d2048,d2048,d" --save-path log/layerres/save/layer3_res_isa > log/layerres/save/layer3_res_isa/train.log
# 
# python train_gcn_basic.py --hidden-layers "d2048,d2048,d2048,d" --no-residual --save-path log/layerres/save/layer4_nores_isa > log/layerres/save/layer4_nores_isa/train.log
# python train_gcn_basic.py --hidden-layers "d2048,d2048,d2048,d" --save-path log/layerres/save/layer4_res_isa > log/layerres/save/layer4_res_isa/train.log
# 
# python train_gcn_basic.py --hidden-layers "d2048,d2048,d2048,d2048,d" --no-residual --save-path log/layerres/save/layer5_nores_isa > log/layerres/save/layer5_nores_isa/train.log
# python train_gcn_basic.py --hidden-layers "d2048,d2048,d2048,d2048,d" --save-path log/layerres/save/layer5_res_isa > log/layerres/save/layer5_res_isa/train.log
# 
# python train_gcn_basic.py --hidden-layers "d2048,d2048,d2048,d2048,d2048,d" --no-residual --save-path log/layerres/save/layer6_nores_isa > log/layerres/save/layer6_nores_isa/train.log
# python train_gcn_basic.py --hidden-layers "d2048,d2048,d2048,d2048,d2048,d" --save-path log/layerres/save/layer6_res_isa > log/layerres/save/layer6_res_isa/train.log
# 
# python train_gcn_relation.py --hidden-layers "d" --no-residual --save-path log/layerres/save/layer1_nores_wup > log/layerres/save/layer1_nores_wup/train.log
# python train_gcn_relation.py --hidden-layers "d" --save-path log/layerres/save/layer1_res_wup > log/layerres/save/layer1_res_wup/train.log
# 
# python train_gcn_relation.py --hidden-layers "d2048,d" --no-residual --save-path log/layerres/save/layer2_nores_wup > log/layerres/save/layer2_nores_wup/train.log
# python train_gcn_relation.py --hidden-layers "d2048,d" --save-path log/layerres/save/layer2_res_wup > log/layerres/save/layer2_res_wup/train.log
# 
# python train_gcn_relation.py --hidden-layers "d2048,d2048,d" --no-residual --save-path log/layerres/save/layer3_nores_wup > log/layerres/save/layer3_nores_wup/train.log
# python train_gcn_relation.py --hidden-layers "d2048,d2048,d" --save-path log/layerres/save/layer3_res_wup > log/layerres/save/layer3_res_wup/train.log
# 
# python train_gcn_relation.py --hidden-layers "d2048,d2048,d2048,d" --no-residual --save-path log/layerres/save/layer4_nores_wup > log/layerres/save/layer4_nores_wup/train.log
# python train_gcn_relation.py --hidden-layers "d2048,d2048,d2048,d" --save-path log/layerres/save/layer4_res_wup > log/layerres/save/layer4_res_wup/train.log
# 
# python train_gcn_relation.py --hidden-layers "d2048,d2048,d2048,d2048,d" --no-residual --save-path log/layerres/save/layer5_nores_wup > log/layerres/save/layer5_nores_wup/train.log
# python train_gcn_relation.py --hidden-layers "d2048,d2048,d2048,d2048,d" --save-path log/layerres/save/layer5_res_wup > log/layerres/save/layer5_res_wup/train.log
# 
# python train_gcn_relation.py --hidden-layers "d2048,d2048,d2048,d2048,d2048,d" --no-residual --save-path log/layerres/save/layer6_nores_wup > log/layerres/save/layer6_nores_wup/train.log
# python train_gcn_relation.py --hidden-layers "d2048,d2048,d2048,d2048,d2048,d" --save-path log/layerres/save/layer6_res_wup > log/layerres/save/layer6_res_wup/train.log
# 
# python evaluate_imagenet.py --pred log/layerres/save/layer1_nores_isa/epoch-1500.pred log/layerres/save/layer1_nores_wup/epoch-1500.pred --test-set 2-hops > log/layerres/layer1_nores_2_hops.txt
# python evaluate_imagenet.py --pred log/layerres/save/layer1_nores_isa/epoch-1500.pred log/layerres/save/layer1_nores_wup/epoch-1500.pred --test-set 3-hops > log/layerres/layer1_nores_3_hops.txt
# python evaluate_imagenet.py --pred log/layerres/save/layer1_nores_isa/epoch-1500.pred log/layerres/save/layer1_nores_wup/epoch-1500.pred --test-set all > log/layerres/layer1_nores_all.txt
# 
# python evaluate_imagenet.py --pred log/layerres/save/layer1_res_isa/epoch-1500.pred log/layerres/save/layer1_res_wup/epoch-1500.pred --test-set 2-hops > log/layerres/layer1_res_2_hops.txt
# python evaluate_imagenet.py --pred log/layerres/save/layer1_res_isa/epoch-1500.pred log/layerres/save/layer1_res_wup/epoch-1500.pred --test-set 3-hops > log/layerres/layer1_res_3_hops.txt
# python evaluate_imagenet.py --pred log/layerres/save/layer1_res_isa/epoch-1500.pred log/layerres/save/layer1_res_wup/epoch-1500.pred --test-set all > log/layerres/layer1_res_all.txt
# 
# python evaluate_imagenet.py --pred log/layerres/save/layer2_nores_isa/epoch-1500.pred log/layerres/save/layer2_nores_wup/epoch-1500.pred --test-set 2-hops > log/layerres/layer2_nores_2_hops.txt
# python evaluate_imagenet.py --pred log/layerres/save/layer2_nores_isa/epoch-1500.pred log/layerres/save/layer2_nores_wup/epoch-1500.pred --test-set 3-hops > log/layerres/layer2_nores_3_hops.txt
# python evaluate_imagenet.py --pred log/layerres/save/layer2_nores_isa/epoch-1500.pred log/layerres/save/layer2_nores_wup/epoch-1500.pred --test-set all > log/layerres/layer2_nores_all.txt
# 
# python evaluate_imagenet.py --pred log/layerres/save/layer2_res_isa/epoch-1500.pred log/layerres/save/layer2_res_wup/epoch-1500.pred --test-set 2-hops > log/layerres/layer2_res_2_hops.txt
# python evaluate_imagenet.py --pred log/layerres/save/layer2_res_isa/epoch-1500.pred log/layerres/save/layer2_res_wup/epoch-1500.pred --test-set 3-hops > log/layerres/layer2_res_3_hops.txt
# python evaluate_imagenet.py --pred log/layerres/save/layer2_res_isa/epoch-1500.pred log/layerres/save/layer2_res_wup/epoch-1500.pred --test-set all > log/layerres/layer2_res_all.txt
# 
# python evaluate_imagenet.py --pred log/layerres/save/layer3_nores_isa/epoch-1500.pred log/layerres/save/layer3_nores_wup/epoch-1500.pred --test-set 2-hops > log/layerres/layer3_nores_2_hops.txt
# python evaluate_imagenet.py --pred log/layerres/save/layer3_nores_isa/epoch-1500.pred log/layerres/save/layer3_nores_wup/epoch-1500.pred --test-set 3-hops > log/layerres/layer3_nores_3_hops.txt
# python evaluate_imagenet.py --pred log/layerres/save/layer3_nores_isa/epoch-1500.pred log/layerres/save/layer3_nores_wup/epoch-1500.pred --test-set all > log/layerres/layer3_nores_all.txt
# 
# python evaluate_imagenet.py --pred log/layerres/save/layer3_res_isa/epoch-1500.pred log/layerres/save/layer3_res_wup/epoch-1500.pred --test-set 2-hops > log/layerres/layer3_res_2_hops.txt
# python evaluate_imagenet.py --pred log/layerres/save/layer3_res_isa/epoch-1500.pred log/layerres/save/layer3_res_wup/epoch-1500.pred --test-set 3-hops > log/layerres/layer3_res_3_hops.txt
# python evaluate_imagenet.py --pred log/layerres/save/layer3_res_isa/epoch-1500.pred log/layerres/save/layer3_res_wup/epoch-1500.pred --test-set all > log/layerres/layer3_res_all.txt

# python evaluate_imagenet.py --pred log/layerres/save/layer4_nores_isa/epoch-1500.pred log/layerres/save/layer4_nores_wup/epoch-1500.pred --test-set 2-hops > log/layerres/layer4_nores_2_hops.txt
# python evaluate_imagenet.py --pred log/layerres/save/layer4_nores_isa/epoch-1500.pred log/layerres/save/layer4_nores_wup/epoch-1500.pred --test-set 3-hops > log/layerres/layer4_nores_3_hops.txt
# python evaluate_imagenet.py --pred log/layerres/save/layer4_nores_isa/epoch-1500.pred log/layerres/save/layer4_nores_wup/epoch-1500.pred --test-set all > log/layerres/layer4_nores_all.txt

# python evaluate_imagenet.py --pred log/layerres/save/layer4_res_isa/epoch-1500.pred log/layerres/save/layer4_res_wup/epoch-1500.pred --test-set 2-hops > log/layerres/layer4_res_2_hops.txt
# python evaluate_imagenet.py --pred log/layerres/save/layer4_res_isa/epoch-1500.pred log/layerres/save/layer4_res_wup/epoch-1500.pred --test-set 3-hops > log/layerres/layer4_res_3_hops.txt
# python evaluate_imagenet.py --pred log/layerres/save/layer4_res_isa/epoch-1500.pred log/layerres/save/layer4_res_wup/epoch-1500.pred --test-set all > log/layerres/layer4_res_all.txt
# 
# python evaluate_imagenet.py --pred log/layerres/save/layer5_nores_isa/epoch-1500.pred log/layerres/save/layer5_nores_wup/epoch-1500.pred --test-set 2-hops > log/layerres/layer5_nores_2_hops.txt
# python evaluate_imagenet.py --pred log/layerres/save/layer5_nores_isa/epoch-1500.pred log/layerres/save/layer5_nores_wup/epoch-1500.pred --test-set 3-hops > log/layerres/layer5_nores_3_hops.txt
# python evaluate_imagenet.py --pred log/layerres/save/layer5_nores_isa/epoch-1500.pred log/layerres/save/layer5_nores_wup/epoch-1500.pred --test-set all > log/layerres/layer5_nores_all.txt
# 
# python evaluate_imagenet.py --pred log/layerres/save/layer5_res_isa/epoch-1500.pred log/layerres/save/layer5_res_wup/epoch-1500.pred --test-set 2-hops > log/layerres/layer5_res_2_hops.txt
# python evaluate_imagenet.py --pred log/layerres/save/layer5_res_isa/epoch-1500.pred log/layerres/save/layer5_res_wup/epoch-1500.pred --test-set 3-hops > log/layerres/layer5_res_3_hops.txt
# python evaluate_imagenet.py --pred log/layerres/save/layer5_res_isa/epoch-1500.pred log/layerres/save/layer5_res_wup/epoch-1500.pred --test-set all > log/layerres/layer5_res_all.txt

python evaluate_imagenet.py --pred log/layerres/save/layer6_nores_isa/epoch-1500.pred log/layerres/save/layer6_nores_wup/epoch-1500.pred --test-set 2-hops > log/layerres/layer6_nores_2_hops.txt
python evaluate_imagenet.py --pred log/layerres/save/layer6_nores_isa/epoch-1500.pred log/layerres/save/layer6_nores_wup/epoch-1500.pred --test-set 3-hops > log/layerres/layer6_nores_3_hops.txt
python evaluate_imagenet.py --pred log/layerres/save/layer6_nores_isa/epoch-1500.pred log/layerres/save/layer6_nores_wup/epoch-1500.pred --test-set all > log/layerres/layer6_nores_all.txt

python evaluate_imagenet.py --pred log/layerres/save/layer6_res_isa/epoch-1500.pred log/layerres/save/layer6_res_wup/epoch-1500.pred --test-set 2-hops > log/layerres/layer6_res_2_hops.txt
python evaluate_imagenet.py --pred log/layerres/save/layer6_res_isa/epoch-1500.pred log/layerres/save/layer6_res_wup/epoch-1500.pred --test-set 3-hops > log/layerres/layer6_res_3_hops.txt
# python evaluate_imagenet.py --pred log/layerres/save/layer6_res_isa/epoch-1500.pred log/layerres/save/layer6_res_wup/epoch-1500.pred --test-set all > log/layerres/layer6_res_all.txt
