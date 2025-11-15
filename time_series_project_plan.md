# TIME SERIES FORECASTING - PROJECT PLAN
## Coffee Shop Revenue Prediction | Senior DS Strategy

**Duration:** 3-4 tuần  
**Level:** Advanced Final Project  
**Business Value:** High - Direct actionable insights

---

## 🎯 PROJECT OVERVIEW

### Business Problem
Coffee shop chain với 3 cửa hàng cần forecast doanh thu để:
- **Inventory optimization:** Giảm waste 10-15%
- **Staff planning:** Right-size labor costs
- **Financial planning:** Better cash flow projection

**Current state:** Manual forecasting dựa vào averages → Inaccurate, miss trends

### Solution
Time series forecasting system với multi-model approach:
- **Target:** Predict daily revenue 7 days ahead
- **Accuracy goal:** MAPE < 15% (retail industry standard)
- **Output:** Automated forecasts + confidence intervals

### Key Success Metrics
| Metric | Target | Business Impact |
|--------|--------|-----------------|
| MAPE | < 15% | Industry benchmark |
| RMSE | < $500 | Acceptable $ error |
| R² | > 0.85 | High explanatory power |
| Forecast horizon | 7 days | Weekly planning cycle |

---

## 📊 DATA ASSESSMENT

### Strengths
- ✅ **Quality cao:** 0 missing values, clean data
- ✅ **Volume:** 149K transactions, 181 days
- ✅ **Strong signal:** 
  - Trend correlation: 0.919 (very strong)
  - Clear weekly seasonality
  - Stable patterns across stores
- ✅ **Rich features:** Time, store, product, price

### Limitations & Risks
- ⚠️ **Short history:** Chỉ 6 tháng → Không có full yearly cycle
- ⚠️ **No external data:** Weather, holidays, events
- ⚠️ **No customer info:** Cannot do customer-level analysis
- ⚠️ **Static pricing:** Không có price variation data

### Mitigation Strategies
1. Focus on short-term forecasts (7-14 days) thay vì long-term
2. Use calendar features as proxy cho seasonality
3. Plan để incorporate external data trong v2.0
4. Monitor model degradation và retrain frequently

---

## 🗓️ PROJECT TIMELINE

### Week 1: Foundation & Baselines (5 days)

**Day 1-2: Deep EDA**
- Aggregate transaction → daily revenue time series
- Decompose: trend + seasonality + residual
- Stationarity tests (ADF, KPSS)
- ACF/PACF analysis → determine ARIMA parameters
- Pattern discovery: hourly, daily, weekly, monthly
- Store-level comparison
- Product category analysis

**Key Deliverables:**
- Time series plots với annotations
- Seasonal decomposition charts
- Statistical test results
- Pattern summary report
- **Decision point:** Confirm forecasting approach

**Day 3: Feature Engineering Design**
- Map out 100+ features:
  - **Temporal:** Calendar features + cyclical encoding
  - **Lag:** 1, 2, 3, 7, 14, 21, 28 days
  - **Rolling:** Mean, std, min, max (windows: 3, 7, 14, 28)
  - **Domain:** Revenue per transaction, volatility, trend
  - **Product mix:** Category revenue breakdown
- Feature selection strategy: Mutual information
- Handle data leakage: Always shift(1) for rolling/lag

**Key Deliverables:**
- Feature engineering blueprint
- Expected features list (~120 features)
- Data leakage prevention checklist

**Day 4-5: Baseline Models**
- Train/Val/Test split: 80/10/10 (TEMPORAL - no shuffle!)
  - Train: Jan 1 - May 24 (144 days)
  - Val: May 25 - Jun 11 (18 days)
  - Test: Jun 12 - Jun 30 (18 days)

- **Simple baselines:**
  - Naive (persistence): Tomorrow = Today
  - Moving average: 3, 7, 14, 28 days
  - Seasonal naive: Tomorrow = Same day last week

- **Classical models:**
  - ARIMA: Grid search (p, d, q)
  - SARIMA: With weekly seasonality (period=7)
  - Auto-ARIMA: Automated parameter selection

**Key Deliverables:**
- 6-8 baseline models trained
- Performance comparison table
- Benchmark set: Best baseline is target to beat
- **Expected:** SARIMA with MAPE ~12-15%

---

### Week 2: Advanced Models (7 days)

**Day 6-7: Gradient Boosting**
- **XGBoost:**
  - Hyperparameter tuning: n_estimators, learning_rate, max_depth
  - CV strategy: TimeSeriesSplit (không dùng regular CV!)
  - Feature importance analysis
  
- **LightGBM:**
  - Early stopping với validation set
  - Compare với XGBoost
  - Speed vs accuracy tradeoff

- **Random Forest:**
  - Baseline tree-based model
  - Feature importance comparison

**Strategy:** Start with default params → Narrow grid search → Final tuning

**Key Deliverables:**
- 3 tree-based models
- Hyperparameter search results
- Feature importance rankings (consistent across models?)
- **Expected:** XGBoost best with MAPE ~8-12%

**Day 8-9: Prophet (Facebook)**
- Why Prophet: 
  - Built for business time series
  - Auto-detects changepoints
  - Handles missing data và outliers
  - Interpretable components

- **Configuration:**
  - Weekly seasonality: TRUE (clear pattern)
  - Yearly seasonality: FALSE (only 6 months data)
  - Holidays: Add US federal holidays
  - Changepoint prior scale: Tune flexibility

- **Extensions:**
  - Add custom regressors (if improve performance)
  - Multiple seasonalities testing

**Key Deliverables:**
- Prophet model với tuned parameters
- Component plots: trend + weekly + residual
- Comparison với tree-based models
- **Expected:** Comparable to XGBoost, better interpretability

**Day 10-11: Ensemble & Advanced Techniques**

**Ensemble strategies:**
1. **Simple average:** Equal weights
2. **Weighted average:** Weights based on validation MAE
3. **Stacking:** Meta-model combines base predictions
4. **Conditional:** Best model per scenario (weekday vs weekend)

**Advanced considerations:**
- **Quantile regression:** Prediction intervals (10%, 50%, 90%)
- **LSTM (Optional):** If time permits và want to show DL
  - Sequence length: 14-28 days
  - Architecture: 2 LSTM layers + dropout
  - Warning: Often overkill cho loại data này
- **Multi-output:** Predict multiple horizons simultaneously (1-7 days)

**Key Deliverables:**
- 2-3 ensemble models
- Comparison: Single vs Ensemble
- Prediction intervals implementation
- **Decision:** Select top 2-3 models for production

---

### Week 3: Production & Validation (7 days)

**Day 12: Comprehensive Evaluation**

**Test set analysis:**
- Multiple metrics: RMSE, MAE, MAPE, R², MBD (bias)
- Temporal breakdown: Performance by week, day of week
- Error analysis: Systematic over/under-prediction?
- Outlier days: Why did model fail?

**Residual diagnostics:**
- Normality tests: Shapiro-Wilk, Q-Q plot
- Autocorrelation: Durbin-Watson (residuals should be random)
- Heteroscedasticity: Breusch-Pagan test
- Pattern in residuals → Missing features?

**Key Deliverables:**
- Complete evaluation report
- Residual analysis plots
- Error breakdown by segments
- Model selection justification doc

**Day 13: Multi-step Ahead Forecasting**

**Challenge:** Forecasting 7 days requires handling uncertainty accumulation

**Approaches:**
1. **Recursive (Iterative):**
   - Predict day 1 → Use as input for day 2 → Repeat
   - Pros: Uses model's own predictions
   - Cons: Errors compound

2. **Direct:**
   - Train 7 separate models (one per horizon)
   - Pros: No error compounding
   - Cons: 7x training time

3. **MIMO (Multi-input Multi-output):**
   - Single model predicts all 7 days at once
   - Pros: Efficient, captures correlations
   - Cons: Complex implementation

**Implementation plan:**
- Test all 3 approaches on validation set
- Compare accuracy degradation by horizon
- Select best approach
- Document uncertainty quantification

**Key Deliverables:**
- Multi-step forecasting implementation
- Horizon-specific accuracy analysis
- Uncertainty intervals (wider for longer horizons)

**Day 14-15: Retraining Pipeline**

**Requirements:**
- **Frequency:** Weekly retraining (as new data arrives)
- **Automation:** No manual intervention required
- **Validation:** Auto-check performance before deployment
- **Rollback:** Keep previous model if new one worse

**Pipeline components:**
1. **Data ingestion:** Pull new transaction data
2. **Feature engineering:** Apply same transformations
3. **Model training:** Use best model from development
4. **Validation:** Check on recent data
5. **Model versioning:** Save with timestamp
6. **Deployment:** Update production model
7. **Monitoring:** Log performance metrics

**Trigger conditions:**
- Scheduled: Every Monday morning
- Performance-based: MAPE > 15% for 3 consecutive days
- Manual: On-demand retraining

**Key Deliverables:**
- Automated training script
- Pipeline orchestration (Airflow/Prefect pattern)
- Model versioning strategy
- Rollback procedure document

**Day 16-17: Monitoring & Alerting**

**Monitoring strategy:**

**1. Performance monitoring:**
- Daily forecast vs actual comparison
- Rolling 7-day MAPE
- Drift detection: Data distribution changes?

**2. Data quality checks:**
- Missing values (should be 0)
- Value ranges (revenue > 0, reasonable bounds)
- Temporal gaps (no missing days)

**3. Model health:**
- Prediction time (latency)
- Memory usage
- API uptime (if applicable)

**Alerting rules:**
| Condition | Severity | Action |
|-----------|----------|--------|
| MAPE > 20% for 1 day | Warning | Log only |
| MAPE > 20% for 3 days | Critical | Alert + auto-retrain |
| MAPE > 30% | Emergency | Alert + manual review |
| Prediction latency > 5s | Warning | Infrastructure check |
| Missing data | Critical | Stop pipeline |

**Dashboard components:**
- Real-time forecast vs actual plot
- Performance metrics (7-day rolling)
- Feature importance tracking (stability check)
- Model version history

**Key Deliverables:**
- Monitoring script/system
- Alert configuration
- Dashboard mockup/implementation
- Incident response playbook

**Day 18: API Development**

**API Requirements:**
- **Endpoint 1:** `/forecast` - Get n-day ahead predictions
- **Endpoint 2:** `/health` - System health check  
- **Endpoint 3:** `/retrain` - Trigger manual retraining (protected)
- **Endpoint 4:** `/metrics` - Get model performance stats

**Design decisions:**
- Framework: FastAPI (fast, modern, auto docs)
- Authentication: API key based
- Rate limiting: Prevent abuse
- Response time: < 1 second
- Caching: Cache recent forecasts (valid for 24h)

**Production considerations:**
- Docker containerization
- Load balancing (if high traffic)
- Logging: Structured logs for debugging
- Error handling: Graceful degradation
- Documentation: OpenAPI/Swagger auto-generated

**Key Deliverables:**
- REST API implementation
- API documentation
- Dockerfile + docker-compose
- Deployment guide (local/cloud)

---

### Week 4: Documentation & Presentation (5 days)

**Day 19: Technical Documentation**

**Documents to create:**

**1. README.md**
- Project overview
- Quick start guide
- Installation instructions
- Usage examples
- Project structure
- Contributing guidelines

**2. Model Card**
- Model details (architecture, version)
- Intended use cases
- Training data description
- Evaluation results
- Limitations
- Ethical considerations
- Monitoring plan

**3. API Documentation**
- Endpoint specifications
- Request/response schemas
- Authentication guide
- Rate limits
- Error codes
- Example calls (curl, Python, JavaScript)

**4. Deployment Guide**
- Infrastructure requirements
- Environment setup
- Configuration
- Deployment steps
- Troubleshooting

**5. Maintenance Manual**
- Retraining schedule
- Monitoring procedures
- Alert response
- Model update process
- Backup and recovery

**Key Deliverables:**
- Complete documentation set
- Code comments cleaned up
- Reproducible examples
- FAQ section

**Day 20: Presentation Preparation**

**Presentation structure (15-20 minutes):**

**1. Business Context (2 min)**
- Problem statement
- Current pain points
- Opportunity size

**2. Data Overview (2 min)**
- Dataset description
- Key statistics
- Data quality

**3. Exploratory Insights (3 min)**
- Trend discovery (92% growth!)
- Seasonality patterns
- Store comparisons
- "Aha" moments from EDA

**4. Modeling Approach (3 min)**
- Why time series forecasting?
- Model selection strategy
- Baseline → Advanced progression
- Feature engineering highlights

**5. Results (4 min)**
- Model comparison table
- Best model performance
- Prediction visualizations
- Error analysis

**6. Business Impact (2 min)**
- Forecast accuracy achieved
- Expected cost savings
- Implementation recommendations
- ROI calculation

**7. Technical Implementation (2 min)**
- Production pipeline overview
- API demonstration
- Monitoring approach

**8. Limitations & Next Steps (2 min)**
- Known limitations (6 months data)
- Future improvements (weather data, hourly)
- Scalability considerations

**Presentation materials:**
- PowerPoint/Google Slides (10-12 slides)
- Live demo notebook (optional)
- Key visualizations:
  - Time series with forecast
  - Model comparison chart
  - Feature importance
  - Actual vs Predicted scatter
  - Residual plots
- Backup slides (technical details)

**Presentation tips:**
- Lead with business value
- Use visualizations > tables
- Tell a story arc
- Practice timing (don't rush)
- Prepare for Q&A

**Key Deliverables:**
- Presentation slides
- Speaker notes
- Demo script (if live demo)
- Backup materials

---

## 🎯 CRITICAL SUCCESS FACTORS

### Technical Excellence
1. **Proper time series handling:**
   - ✅ Temporal train/test split (NEVER shuffle!)
   - ✅ Feature engineering avoids data leakage
   - ✅ Walk-forward validation
   - ✅ Horizon-specific evaluation

2. **Model rigor:**
   - ✅ Multiple baselines for comparison
   - ✅ Hyperparameter tuning with proper CV
   - ✅ Ensemble methods considered
   - ✅ Uncertainty quantification (prediction intervals)

3. **Production readiness:**
   - ✅ Automated retraining pipeline
   - ✅ Monitoring and alerting
   - ✅ API for easy access
   - ✅ Documentation complete

### Project Management
1. **Scope control:**
   - Focus on daily revenue first
   - Hourly/product-level = stretch goals
   - Don't over-engineer v1.0

2. **Time management:**
   - Week 1: Must finish baselines
   - Week 2: Advanced models done
   - Week 3: Production pipeline critical
   - Week 4: Buffer for polish

3. **Risk mitigation:**
   - Start simple, add complexity
   - Have checkpoint demos
   - Document decisions
   - Keep old working versions

---

## 📈 EXPECTED OUTCOMES

### Model Performance (Realistic Targets)

**Conservative scenario:**
- Best model: SARIMA or XGBoost
- MAPE: 12-15%
- RMSE: $450-550
- R²: 0.80-0.85

**Optimistic scenario:**
- Best model: XGBoost Ensemble
- MAPE: 8-12%
- RMSE: $350-450
- R²: 0.85-0.90

**Model ranking (expected):**
1. 🥇 XGBoost/LightGBM (best accuracy)
2. 🥈 Prophet (best interpretability)
3. 🥉 SARIMA (best for pure time series)
4. Random Forest (good baseline)
5. ARIMA (baseline)
6. Naive/MA (simple baselines)

### Business Value Quantification

**Scenario analysis:**

**Baseline (current state):**
- Inventory waste: 15% due to over/under-stocking
- Labor cost inefficiency: 10% due to poor scheduling
- Annual revenue: ~$1.4M (extrapolated from 6 months)

**With forecasting (target state):**
- Inventory waste reduced to 8% (save 7 percentage points)
- Labor efficiency improved (save 5%)
- Better cash flow planning (intangible)

**Estimated savings:**
- Inventory: $1.4M × 7% = $98,000/year
- Labor: $350K (est. labor cost) × 5% = $17,500/year
- **Total:** ~$115K/year

**Implementation cost:**
- DS time: 3-4 weeks
- Infrastructure: ~$100/month (API hosting)
- Maintenance: 2-4 hours/week

**ROI:** Positive in < 2 months

---

## 🚨 COMMON PITFALLS & HOW TO AVOID

### Pitfall 1: Data Leakage
**Problem:** Using future information in features
**Solution:** 
- Always shift(1) for lag and rolling features
- Validate train/test split is temporal
- Check feature creation dates carefully

### Pitfall 2: Wrong CV Strategy
**Problem:** Using random CV on time series
**Solution:**
- Use TimeSeriesSplit or walk-forward validation
- Respect temporal order always
- Never shuffle time series data

### Pitfall 3: Overfitting to Recent Data
**Problem:** Model works great on validation, fails on test
**Solution:**
- Multiple validation windows
- Test on unseen time period
- Regularization in tree models

### Pitfall 4: Ignoring Uncertainty
**Problem:** Point forecasts without confidence intervals
**Solution:**
- Quantile regression for intervals
- Monte Carlo simulation
- Bayesian approaches (if time)

### Pitfall 5: Over-engineering
**Problem:** Spending too much time on minor improvements
**Solution:**
- Follow 80/20 rule
- Focus on business value first
- Complexity adds later

### Pitfall 6: Poor Documentation
**Problem:** Nobody can use/maintain the model
**Solution:**
- Document as you go (not at end)
- Code comments + docstrings
- README with examples

---

## 🛠️ TECHNOLOGY STACK

### Core Libraries
- **Data:** pandas, numpy
- **Viz:** matplotlib, seaborn, plotly
- **Stats:** statsmodels (ARIMA, diagnostics)
- **ML:** scikit-learn, xgboost, lightgbm
- **Time Series:** prophet (Facebook)
- **Deep Learning (optional):** tensorflow/pytorch

### Production
- **API:** FastAPI
- **Containerization:** Docker
- **Orchestration:** Airflow/Prefect (or simple cron)
- **Monitoring:** Custom scripts + logging
- **Storage:** pickle/joblib for models

### Development
- **Environment:** Jupyter notebooks + Python scripts
- **Version control:** Git
- **Experiment tracking:** MLflow (optional but recommended)

---

## 📚 LEARNING OUTCOMES

By completing this project, you will demonstrate:

### Technical Skills
- ✅ Time series analysis và forecasting
- ✅ Feature engineering for temporal data
- ✅ Model selection và validation
- ✅ Ensemble methods
- ✅ Production ML pipeline development
- ✅ API development
- ✅ Monitoring và maintenance

### Soft Skills
- ✅ Problem definition (business → technical)
- ✅ Project planning và time management
- ✅ Communication (technical → non-technical)
- ✅ Documentation
- ✅ Presentation skills

### Portfolio Value
- 🌟 End-to-end project (EDA → Deployment)
- 🌟 Production-ready code
- 🌟 Business value quantification
- 🌟 Multiple modeling techniques
- 🌟 Best practices demonstrated

---

## 📖 RECOMMENDED RESOURCES

### Must-Read
1. "Forecasting: Principles and Practice" - Rob Hyndman (free online)
2. "Time Series Analysis" - Box, Jenkins, Reinsel
3. Facebook Prophet documentation
4. XGBoost for Time Series (blog posts)

### Helpful
- Kaggle time series competitions
- Medium articles on production ML
- FastAPI documentation
- MLOps best practices

### Tools to Explore
- MLflow for experiment tracking
- Weights & Biases for monitoring
- Evidently AI for model monitoring
- Great Expectations for data validation

---

## ✅ FINAL CHECKLIST

### Before Starting
- [ ] Data accessible và validated
- [ ] Development environment setup
- [ ] Git repository initialized
- [ ] Project structure created
- [ ] Timeline reviewed với advisor/team

### Week 1 Completion
- [ ] EDA complete với insights documented
- [ ] Feature engineering pipeline working
- [ ] Baseline models trained
- [ ] Performance benchmark established

### Week 2 Completion
- [ ] Advanced models trained
- [ ] Model comparison done
- [ ] Top 2-3 models selected
- [ ] Feature importance analyzed

### Week 3 Completion
- [ ] Final model selected với justification
- [ ] Multi-step forecasting working
- [ ] Retraining pipeline automated
- [ ] API deployed

### Week 4 Completion
- [ ] All documentation complete
- [ ] Presentation ready
- [ ] Code cleaned và commented
- [ ] Repository organized
- [ ] Demo tested

### Presentation Day
- [ ] Slides polished
- [ ] Demo rehearsed
- [ ] Timing practiced
- [ ] Q&A prepared
- [ ] Backup materials ready

---

## 🎓 CONCLUDING THOUGHTS

**Why This Project is Excellent for Final Assignment:**

1. **Real business value:** Not just academic exercise
2. **Full DS lifecycle:** From EDA to deployment
3. **Technical depth:** Multiple modeling approaches
4. **Production focus:** Not just notebooks
5. **Clear metrics:** Easy to evaluate success
6. **Scalable:** Can add features incrementally
7. **Portfolio-worthy:** Shows end-to-end capability

**What Makes This Stand Out:**
- Strong upward trend in data → Clear story
- Multiple modeling approaches → Shows versatility
- Production pipeline → Shows engineering skills
- Business impact quantification → Shows business acumen
- Complete documentation → Shows professionalism

**Key Success Factors:**
1. Start simple, add complexity progressively
2. Document decisions and trade-offs
3. Focus on business value throughout
4. Test everything thoroughly
5. Present clearly and confidently

Good luck! 🚀

---

**Last Updated:** November 2025  
**Version:** 1.0  
**Contact:** [Your information]
