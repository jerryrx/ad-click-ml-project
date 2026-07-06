# Group 4 — Ad Click Prediction (South Africa — Adbot)

## The Business Problem

South Africa has thousands of small and medium-sized businesses trying to
grow through online advertising. Most do not have the expertise or budget
to hire a marketing team. They rely on platforms like Adbot to manage
their Google ads automatically.

The key question for every business owner is simple: how many people will
click on my ad? Clicks drive website visits, phone calls, and ultimately
sales. If a business can predict how many clicks an ad will get, they can
decide how much to spend, which keywords to use, and when to run their ads.

## The Solution

Build a regression model that predicts the number of clicks an ad record
will receive based on the ad's characteristics — impressions, cost,
ad type, headline length, description length, and call engagement metrics.

## The Business Value

- Small businesses can forecast ad performance before committing budget
- Adbot can automatically optimise campaigns based on predicted clicks
- Businesses allocate advertising budgets more efficiently
- Underperforming ads are identified early and replaced
- ROI on advertising spending improves across all clients

## The Data

Daily ad performance records from 185 small and medium business clients
on the Adbot platform in South Africa. Data spans from January 2020 to
February 2024 - over 4 years of real advertising data.


## Target Variable

`clicks` — the number of times someone clicked on an advertisement on a given day.

Statistics: min=0, max=4,227, mean=8.04, median=2
Right-skewed distribution — most ads get few clicks but some get very many.

## Column Descriptions

### Ad Performance Metrics

| Column | Description |
|---|---|
| `impressions` | Number of times the ad was shown to users |
| `clicks` | Number of times someone clicked the ad — TARGET VARIABLE |
| `cost` | Amount spent on the ad in ZAR or USD |
| `conversions` | Number of times a click led to a desired action (sale, form fill, etc.) |
| `impression_share` | Percentage of auctions where the ad was shown vs total eligible |
| `conversions_calls` | Number of conversions that came through phone calls |

### Ad Characteristics

| Column | Description |
|---|---|
| `ad_type` | Type of ad: EXPANDED_TEXT_AD, RESPONSIVE_SEARCH_AD, EXPANDED_DYNAMIC_SEARCH_AD |
| `headline1_len` | Number of words in the first headline of the ad |
| `headline2_len` | Number of words in the second headline of the ad |
| `ad_description_len` | Number of words in the ad description |

### Campaign and Client

| Column | Description |
|---|---|
| `ID` | Unique client business identifier (185 unique clients) |
| `currency` | Currency of the ad spend — ZAR (South African Rand) or USD |
| `date` | Date of the ad record |

### Call Data (from clients who use call tracking)

| Column | Description |
|---|---|
| `call_type` | Type of call received (many missing — not all clients have call tracking) |
| `call_status` | Whether the call was Received or Missed |
| `start_time` | Time the call started |
| `end_time` | Time the call ended |
| `duration` | Duration of the call in seconds |
| `display_location` | Where the ad was displayed |

## Key Challenges

- Right-skewed target: most ads get 0-10 clicks but some get thousands
- call_type, call_status, start_time, end_time have 87% missing — drop these
- 185 different clients with very different click volumes
- date needs to be parsed to extract day of week, month, year
- impressions is strongly correlated with clicks — this is expected

## Evaluation Metric

RMSE (Root Mean Squared Error) — measures average prediction error in clicks.
Also report MAE and R² for a complete picture.

## Suggested Approach

1. Drop high-missing columns: call_type, call_status, start_time, end_time, display_location
2. Parse date to extract: day of week, month, year
3. Fill missing numerical values with median
4. Encode ad_type and currency
5. Train and compare: Linear Regression, Decision Tree, Random Forest, XGBoost, LightGBM
6. Tune with RandomizedSearchCV scoring='neg_root_mean_squared_error'
7. Note: impressions is the strongest predictor — make sure students understand why
