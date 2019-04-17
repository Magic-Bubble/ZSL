
python evaluate_imagenet.py --pred save/results/gcn-isa-residual-epoch-1500.2502.pred save/results/gcn-wup-residual-epoch-1500.2429.pred --test-set 2-hops > log/basic/RGCN-DKGs-2-hops.txt

python evaluate_imagenet.py --pred save/results/gcn-isa-residual-epoch-1500.2502.pred save/results/gcn-wup-residual-epoch-1500.2429.pred --test-set 3-hops > log/basic/RGCN-DKGs-3-hops.txt

python evaluate_imagenet.py --pred save/results/gcn-isa-residual-epoch-1500.2502.pred save/results/gcn-wup-residual-epoch-1500.2429.pred --test-set all > log/basic/RGCN-DKGs-all.txt

python evaluate_imagenet.py --pred save/results/gcn-isa-residual-epoch-1500.2502.pred save/results/gat-wup-residual-epoch-900.2462.pred --test-set 2-hops > log/basic/ARGCN-DKGs-2-hops.txt

python evaluate_imagenet.py --pred save/results/gcn-isa-residual-epoch-1500.2502.pred save/results/gat-wup-residual-epoch-900.2462.pred --test-set 3-hops > log/basic/ARGCN-DKGs-3-hops.txt

python evaluate_imagenet.py --pred save/results/gcn-isa-residual-epoch-1500.2502.pred save/results/gat-wup-residual-epoch-900.2462.pred --test-set all > log/basic/ARGCN-DKGs-all.txt
