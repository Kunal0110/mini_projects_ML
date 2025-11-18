# Customer Segmentation Report

## 1. Introduction
This report presents a customer segmentation analysis using the Mall Customers dataset. The objective is to identify distinct customer groups based on age, income, and spending behaviour, and to derive actionable insights for targeted marketing.

## 2. Data Overview
- Number of customers: 200
- Features used: Age, Annual Income (k$), Spending Score (1-100)

## 3. Methodology
- Preprocessing: Selected numeric features and applied standardization.
- Clustering algorithm: K-means with k = 6 clusters.
- Evaluation: Number of clusters chosen based on elbow and silhouette methods (see notebook for details).

## 4. Cluster Profiles
### Cluster 0
- Number of customers: 45.0
- Average age: 56.3
- Average annual income (k$): 54.3
- Average spending score (1-100): 49.1
- Description: _Add a human-friendly label here (e.g., 'Young high spenders')._

### Cluster 1
- Number of customers: 39.0
- Average age: 26.8
- Average annual income (k$): 57.1
- Average spending score (1-100): 48.1
- Description: _Add a human-friendly label here (e.g., 'Young high spenders')._

### Cluster 2
- Number of customers: 33.0
- Average age: 41.9
- Average annual income (k$): 88.9
- Average spending score (1-100): 17.0
- Description: _Add a human-friendly label here (e.g., 'Young high spenders')._

### Cluster 3
- Number of customers: 39.0
- Average age: 32.7
- Average annual income (k$): 86.5
- Average spending score (1-100): 82.1
- Description: _Add a human-friendly label here (e.g., 'Young high spenders')._

### Cluster 4
- Number of customers: 23.0
- Average age: 25.0
- Average annual income (k$): 25.3
- Average spending score (1-100): 77.6
- Description: _Add a human-friendly label here (e.g., 'Young high spenders')._

### Cluster 5
- Number of customers: 21.0
- Average age: 45.5
- Average annual income (k$): 26.3
- Average spending score (1-100): 19.4
- Description: _Add a human-friendly label here (e.g., 'Young high spenders')._

## 5. Actionable Insights
Below are example types of strategies the mall could adopt for different clusters. You should refine these based on the exact profiles observed:

- High-income, low-spending clusters: focus on premium experiences and personalized offers.
- Young, high-spending clusters: emphasize fashion, entertainment, and digital campaigns.
- Budget-conscious clusters: promote discounts, bundles, and loyalty points.

## 6. Conclusion
The analysis highlights distinct segments within the mall's customer base. By tailoring marketing campaigns and offerings to each segment, the mall can increase engagement, optimize promotional budgets, and improve overall customer satisfaction.
