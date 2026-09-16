# Project Report: Uber Trip Analytics & Demand Insights

## 1. Introduction
Ride-hailing services generate massive volumes of trip data every day. Understanding *when* and *where* demand occurs is critical for operational planning (driver allocation, surge pricing, marketing). This project performs an exploratory data analysis (EDA) of Uber pickup records in New York City between April and September 2014, using Python's data science stack: `pandas`, `NumPy`, `Matplotlib`, and `Seaborn`.

## 2. Problem Statement
Given ~4.5 million raw, timestamped Uber pickup records (latitude, longitude, timestamp, dispatch base), identify and visualize:
- Temporal demand patterns (by hour, day, weekday, month)
- Demand distribution across the five NYC dispatch bases
- Geographic distribution of pickups across New York City

## 3. Tools & Technologies
- **Language:** Python 3
- **Libraries:** pandas, NumPy, Matplotlib, Seaborn

## 4. Dataset Description
- Source: NYC Uber trip logs (Apr–Sep 2014), 6 CSV files (one per month)
- Fields: `Date/Time`, `Lat`, `Lon`, `Base`
- Combined size: ~4.5 million rows

## 5. Methodology
1. **Data Import** — read six monthly CSVs with `pandas.read_csv()` and combine with `pd.concat()` into one DataFrame.
2. **Data Cleaning & Feature Engineering** — parse `Date/Time` with `pd.to_datetime()`; derive `hour`, `minute`, `second`, `day`, `month`, `dayofweek` using pandas' `.dt` accessor. Month and weekday are cast to ordered categoricals so charts display in calendar order rather than alphabetical order.
3. **Aggregation** — use `DataFrame.groupby()` + `.size()` (equivalent to a SQL `GROUP BY ... COUNT(*)`) to compute trip counts across each dimension (hour, day, month, weekday, base) and combinations of dimensions. `.unstack()` is used to pivot grouped results into a wide format suitable for stacked bar charts and heatmaps.
4. **Visualization**:
   - **Matplotlib** bar charts for single-dimension aggregations (trips by hour, by day, by month, by base) and the geospatial scatter plot.
   - **Seaborn** grouped bar charts (`sns.barplot` with `hue=`) for two-dimension comparisons (e.g., month × weekday, base × month).
   - **Seaborn heatmaps** (`sns.heatmap`) for two-dimensional cross-tabulations (Hour×Day, Month×Day, Month×Weekday, Base×Month, Base×Weekday).
   - A **geospatial scatter plot** of `Lat`/`Lon` coordinates, cropped to NYC's bounding box, colored by dispatch base.

## 6. Results / Key Findings
- Ride volume peaks in the **evening (5–6 PM)**, consistent with commuter rush hour.
- The **30th of the month** recorded the highest single-day trip count (largely due to April).
- **September** was the busiest month overall.
- Dispatch base **B02617** generated the highest trip volume across the period.
- **Thursday** was the peak weekday for the three busiest bases (B02598, B02617, B02682).
- Geographically, pickups are heavily concentrated in **Manhattan**, with visible clustering near major hubs.

*(Run `scripts/uber_analysis.py` against the full dataset to reproduce and verify these exact figures.)*

## 7. Challenges Faced
- **Memory/performance**: ~4.5 million rows requires adequate RAM; `groupby` operations on the full dataset are noticeably slower than on samples — chained/vectorized pandas operations were used instead of Python loops to keep this efficient.
- **Category ordering**: pandas orders string categories alphabetically by default (e.g., "Apr, Aug, Jul..."), which is misleading for a calendar sequence. This was fixed by explicitly casting `month` and `dayofweek` to ordered `Categorical` types.
- **Geospatial cropping**: rather than letting outlier GPS points stretch the axis range, the geospatial plot is explicitly filtered and limited to NYC's known bounding box (`Lat`: 40.5774–40.9176, `Lon`: -74.15– -73.7004).

## 8. Conclusion
This project demonstrates a complete, real-world EDA workflow in Python: from ingesting raw multi-file data to deriving temporal features and producing clear, decision-useful visualizations. It highlights how visualization (rather than raw tables) makes large-scale operational data interpretable — a skill directly transferable to any data analytics role.

## 9. Future Scope
- Extend analysis to more recent/larger Uber datasets.
- Build an interactive Streamlit or Plotly Dash dashboard for live exploration.
- Correlate demand spikes with external factors (weather, holidays, events).
- Add a time-series demand forecasting model as a follow-up project.
