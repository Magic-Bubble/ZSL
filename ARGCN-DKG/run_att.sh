
python evaluate_imagenet.py --pred save/results/gcn-isa-residual-epoch-1500.2502.pred --test-set all > log/att/isa-noatt-all.txt
python evaluate_imagenet.py --pred save/results/gat-isa-residual-epoch-900.2483.pred --test-set all > log/att/isa-att-all.txt
python evaluate_imagenet.py --pred save/results/gcn-wup-residual-epoch-1500.2429.pred --test-set all > log/att/wup-noatt-all.txt
python evaluate_imagenet.py --pred save/results/gat-wup-residual-epoch-900.2462.pred --test-set all > log/att/wup-att-all.txt

python evaluate_imagenet.py --pred save/results/gcn-isa-residual-epoch-1500.2502.pred --test-set 3-hops > log/att/isa-noatt-3-hops.txt
python evaluate_imagenet.py --pred save/results/gat-isa-residual-epoch-900.2483.pred --test-set 3-hops > log/att/isa-att-3-hops.txt
python evaluate_imagenet.py --pred save/results/gcn-wup-residual-epoch-1500.2429.pred --test-set 3-hops > log/att/wup-noatt-3-hops.txt
python evaluate_imagenet.py --pred save/results/gat-wup-residual-epoch-900.2462.pred --test-set 3-hops > log/att/wup-att-3-hops.txt

python evaluate_imagenet.py --pred save/results/gcn-isa-residual-epoch-1500.2502.pred --test-set 2-hops > log/att/isa-noatt-2-hops.txt
python evaluate_imagenet.py --pred save/results/gat-isa-residual-epoch-900.2483.pred --test-set 2-hops > log/att/isa-att-2-hops.txt
python evaluate_imagenet.py --pred save/results/gcn-wup-residual-epoch-1500.2429.pred --test-set 2-hops > log/att/wup-noatt-2-hops.txt
python evaluate_imagenet.py --pred save/results/gat-wup-residual-epoch-900.2462.pred --test-set 2-hops > log/att/wup-att-2-hops.txt
