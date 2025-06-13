import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from preprocess import preprocess_data
from score import load_model, get_feature_importance, predict_scores

def main():
    os.makedirs('input', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    input_file = 'input/test.csv'
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found")
    
    test_data = pd.read_csv(input_file)
    processed_data = preprocess_data(test_data, is_train=False)
    
    model = load_model('catboost_model.cbm')
    predictions = predict_scores(model, processed_data)
    
    submission = pd.DataFrame({'id': range(len(predictions)), 'target': predictions})
    submission.to_csv('output/submission.csv', index=False)
    
    feature_importance = get_feature_importance(model)
    with open('output/feature_importance.json', 'w') as f:
        json.dump(feature_importance, f, indent=4)
    
    plt.figure(figsize=(10, 6))
    plt.hist(predictions, bins=50, density=True)
    plt.title('Distribution of Predicted Scores')
    plt.xlabel('Predicted Score')
    plt.ylabel('Density')
    plt.savefig('output/prediction_distribution.png')
    plt.close()

if __name__ == "__main__":
    main() 