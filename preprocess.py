import pandas as pd

def preprocess_data(df, is_train=True):
    df['transaction_time'] = pd.to_datetime(df['transaction_time'])
    df['hour'] = df['transaction_time'].dt.hour
    df['day'] = df['transaction_time'].dt.day
    df['month'] = df['transaction_time'].dt.month
    df['day_of_week'] = df['transaction_time'].dt.dayofweek
    df['week_of_year'] = df['transaction_time'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['transaction_time'].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    df = df.drop(columns=['transaction_time'])

    categorical_features = [
        'merch', 'cat_id', 'name_1', 'name_2', 'gender',
        'street', 'one_city', 'us_state', 'post_code', 'jobs'
    ]

    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df 