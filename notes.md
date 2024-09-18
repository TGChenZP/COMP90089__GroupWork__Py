- one hot encoding

- The Mann-Whitney U test and Chi-square test was conducted to assess whether the distribution of a particular predictive feature (continuous and categorical, respectively) was significantly different for individuals with and without IA symptoms. 
    - F test (= fit lr with intercept and only one variable)
- spearmanns correlation heatmap between vars:
    - with only one feature in each significantly correlated pair (r > 0.8) retained.
        - if have to drop one of two, drop the one with more mising values (see below)
- if drop these columns, then go back and loop over 
    item_51006 693
    item_51221 719
    item_51222 706
    item_51265 714
    item_51301 705
    sodium 370
    potassium 384
    creatinine 308
    bun 292
    urine_output_total 175
    pt 530
    inr 531
    (temperature 48)
    glucose 168


LAST THING BEFORE EXPORT TO RON

- outlier drop row (?? Ron will ask) 3q + 3iqr, 1q-3iqr 




LOW PRIORITY
- descriptive statistics (mean, var)
