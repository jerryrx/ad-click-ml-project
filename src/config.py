import os

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'adbot_data_90.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pkl')

TARGET_COL   = 'clicks'
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# Drop columns with 87%+ missing or not useful for prediction
DROP_COLS = [
    'call_type', 'call_status', 'start_time',
    'end_time', 'display_location', 'date'
]

CATEGORICAL_COLS = ['ad_type', 'currency', 'ID']

NUMERICAL_COLS = [
    'impressions', 'cost', 'conversions', 'impression_share',
    'conversions_calls', 'headline1_len', 'headline2_len',
    'ad_description_len', 'duration', 'day_of_week', 'month', 'year'
]

MODEL_PARAMS = {
    'n_estimators':  200,
    'learning_rate': 0.05,
    'num_leaves':    31,
    'random_state':  RANDOM_STATE,
    'verbose':       -1
}
