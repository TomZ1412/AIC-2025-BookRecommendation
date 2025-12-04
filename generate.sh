echo "Generating submission for preliminary phase..."
python generate_submission.py --model_file saved/BPR-Nov-21-2025_18-57-05.pth --submission_file generated/submission_pre.csv 
echo "Generating submission for reevaluation phase..."
python generate_submission.py --model_file saved/BPR-Nov-21-2025_18-59-07.pth --submission_file generated/submission_semi.csv 
echo "Generating submission for final phase..."
python generate_submission.py --model_file saved/BPR-Nov-21-2025_19-00-30.pth --submission_file generated/submission_final.csv 
echo "Merging submission files..."
python merge.py
echo "Generating completed!"