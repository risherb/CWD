# Report for CWD

**1 Z-Threshold vs Confidence Score**

**Z-Threshold**

This is the limit of the Z-Score. It is like a boundary the Z-Score must not cross. This is purely based on the amount of the transaction. Nothing else.

**Confidence Score**

Confidence score is a percentage from 0 to 100. It looks at two things together, the transaction amount and the time at which the transaction happened. If a transaction happens outside business hours the confidence scoring metric goes up heavily because of our logic that is used in calculating the score. If the amount is also high then it goes up further.

**Z-Threshold vs Confidence Score**

Z-Threshold is a hard numerical check on the amount alone. Either the Z-Score crosses the limit or it does not. Confidence score on the other hand takes into account both the amount and the timing of the transaction and gives a weighted percentage. Z-Threshold catches outliers purely by how large the amount is. Confidence score catches outliers by looking at the full picture. Both are running independently on every transaction.

**NOTE : In  UI the change in confidence is actually changing the threshold not the confidence, for clarity.**

**2 When Does a Transaction Get Flagged?**

A transaction is marked as anomaly if any one of these is true:

Z-Score of the transaction crosses Z_Threshold

If the timezone is out of effective opening or closing time.

Both do not need to fail. One is enough. If the device has less than 100 transactions in history then the result is marked as "Unsupported" because there is not enough data to judge fairly.

**3 Important Note**

Formula to Calculate Anomaly Threshold Amount:

This is the amount above which any transaction will get flagged:

```
Anomaly Amount = Mean + (Z_Threshold × Standard Deviation)
```

Working Example

Suppose for a device:

Mean transaction amount is ₹1,000

Standard Deviation is ₹200

Z_Threshold is 52

```
Anomaly Amount = 1000 + (52 × 200) = ₹11,400
```

So any transaction above ₹11,400 on this device will be flagged as an anomaly.

---

**4 The difference between Anomaly and Fraud as a transaction status.**

Fraud is a confirmed malicious or unauthorised transaction. An anomaly is simply a significant outlier in the data. It does not mean the transaction is fraudulent. It means it is unusual enough to warrant a closer look. This is why a review section has been added to the system, so that flagged transactions can be further examined and classified as fraud only if there is sufficient basis to do so.

---

**5 More logs in both the AI backend and anomaly detection.**

Logs have been added. An environment variable has been created to enable or disable logging as needed without any changes to the code. Initialization service logs added as well.

