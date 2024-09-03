# Forecast Review for Anticipatory Action (AA) - 2024

### Purpose
The Jupyter Notebook aims to support the forecast review.

### Audience
Tailored for government officials in agriculture, water resources, and disaster management sectors.

### Objectives
To provide an in-depth analysis of the forecast using the Design Tool API.

### Forecast Analysis

The Jupyter Notebook analyzes and visualizes critical metrics across different frequencies for each administrative name within a dataset. These metrics help evaluate the performance of the forecast for decision-making. Here's a summary of what each metric represents and how they are derived:

| Observed             | Drought                                     | No Drought                                       |                                   |
|----------------------|---------------------------------------------|--------------------------------------------------|-----------------------------------|
| **Forecast**         |                                             |                                                  |                                   |
| Drought              | a = hit (Worthy Action)                     | b = false alarm (Act in Vain)                    | a + b = forecast drought          |
|                      |                                             |                                                  |                                   |
| No Drought           | c = missed (Fail to Act)                    | d = correct rejection (Worthy Inaction)          | c + d = forecast no drought       |
|                      |                                             |                                                  |                                   |
|                      | a + c = observed drought                    | b + d = observed no drought                      |                                   |

#### Metric 1: Hit rate (HR)
The hit rate (HR) index represents the number of hits divided by the total number of events observed and therefore describes the fraction of the observed drought events that were correctly forecast. The HR index can range from 0 to 1 (perfect score), and extracted as follow:

$$
HR = \frac{a}{a + c}
$$

Where:
- \( a \) = hits (Worthy Action)
- \( c \) = misses (Fail to Act)

(1)

#### Metric 2: False alarm ratio (FAR)
The False Alarm Ratio is a measure of the proportion of non-drought events per total number of drought warnings (Equation (2)). In drought seasonal forecasting it is important to have a certain balance between over reporting a potential drought versus the pitfall of not issuing an alarm when a drought can in fact happen. A low FAR is preferable, but the decision-maker should have in mind the socio-economic implications of missing an actual drought event.

$$
FAR = \frac{b}{a + b}
$$

Where:
- \( b \) = false alarms (Act in Vain)
- \( a \) = hits (Worthy Action)

(2)

#### Metric 3: Bias score (BS)
The Bias score identifies whether the forecast system tends to under forecast (BS < 1) or over forecast (BS > 1) the occurrence of drought events. In other words, the bias score answers the question of how did the forecasted frequency of droughts events compare to the observed frequency of drought events? The ideal value of the bias score is 1. This metric can be derived applying the following equation:

$$
BS = \frac{a + b}{a + c}
$$

Where:
- \( a \) = hits (Worthy Action)
- \( b \) = false alarms (Act in Vain)
- \( c \) = misses (Fail to Act)
(3)

#### Metric 4: Hanssen-Kuipers score (KSS)
The Hanssen-Kuipers score, also known as the true skill statistic, measures the difference between the hit rate and the false alarm rate (Equation (4)). The KSS measures the ability of the forecast to distinguish between occurrences and non-occurrences of droughts (Mason, 2018). The Hanssen-Kuipers Score ranges from −1 to 1.


$$
KSS = HR - FAR = \frac{ad - bc}{(a + c)(b + d)}
$$

Where:
- \( a \) = hits (Worthy Action)
- \( b \) = false alarms (Act in Vain)
- \( c \) = misses (Fail to Act)
- \( d \) = correct rejections (Worthy Inaction)

(4)

#### Metric 5: Heidke skill score (HSS)
The Heidke Skill score measures the fraction of correct forecasts after eliminating those forecasts which would be correct due to random chance (Mason, 2018). The HSS index can describe the accuracy of the forecast relative to that of random chance ranging from minus infinite to 1 (Equation (5)). A negative value indicates that a forecast is worse than a guess, whereas 0 and 1 values indicate no and perfect skill, respectively.

$$
HSS = \frac{2(ad - bc)}{(a + c)(c + d) + (a + b)(b + d)}
$$

Where:
- \( a \) = hits (Worthy Action)
- \( b \) = false alarms (Act in Vain)
- \( c \) = misses (Fail to Act)
- \( d \) = correct rejections (Worthy Inaction)
(5)

#### Classification of Forecast Accuracy Based on HR and FAR

The performance of the forecasts is classified based on two metrics: Hit Rate (HR) and False Alarm Ratio (FAR). These classifications help in understanding the accuracy and reliability of the forecasts by categorizing them into three categories: 'Good', 'Moderate', and 'Bad'. This classification is critical for decision-making processes where the cost of false positives and false negatives can be significant.

##### Classification Criteria
The classification into 'Good', 'Moderate', and 'Bad' is determined by comparing the values of HR and FAR against each other and a specified threshold. The threshold is set to a default value of 0.6 but can be adjusted based on specific requirements. The classification criteria are as follows:

1. Good: The forecast is classified as 'Good' if the HR is greater than both the FAR and the threshold. This implies a high accuracy in predicting true events with few false alarms.
1. Moderate: The forecast is classified as 'Moderate' if the HR is greater than or equal to the FAR but less than the threshold. This indicates that the forecast system is reasonably accurate but not robust enough to meet the threshold criteria.
1. Bad: The forecast is classified as 'Bad' if the FAR is greater than the HR, suggesting that the forecast system has a higher rate of false alarms compared to correct predictions.


### IMPORTANT - DISCLAIMER AND RIGHTS STATEMENT

This is a set of scripts written by the Financial Instruments Team at The National Center for Disaster Preparedness (NCDP), Columbia Climate School, at Columbia University. They are shared for educational purposes only.  Anyone who uses this code or its functionality or structure assumes full liability and should inform and credit NCDP.

Source - [Forecasting, thresholds, and triggers: Towards developing a Forecast-based Financing system for droughts in Mozambique](https://www.sciencedirect.com/science/article/pii/S2405880723000055)