echo "Training BPR model for preliminary phase..."
python run_recbole.py --phase pre
echo "Training BPR model for reevaluation phase..."
python run_recbole.py --phase semi
echo "Training BPR model for final phase..."
python run_recbole.py --phase final
echo "Training completed!"