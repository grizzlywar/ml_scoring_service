from catboost import CatBoostClassifier, Pool

def load_model(model_path):
    return CatBoostClassifier().load_model(model_path)

def get_feature_importance(model):
    feature_importance = model.get_feature_importance()
    feature_names = model.feature_names_
    
    importance_dict = dict(zip(feature_names, feature_importance))
    top_5 = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5])
    
    return top_5

def predict_scores(model, data):
    categorical_features = [
        'merch', 'cat_id', 'name_1', 'name_2', 'gender',
        'street', 'one_city', 'us_state', 'post_code', 'jobs'
    ]
    
    pool = Pool(data, cat_features=categorical_features)
    return model.predict(pool) 