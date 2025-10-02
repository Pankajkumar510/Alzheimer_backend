# Cognitive Test Question Improvements

## 🎯 Changes Made

### 1. **Added Helper Descriptions**
Every complex question now includes a clear description in a blue info box below the question.

### 2. **Simplified Medical Terms**

| Original | Improved |
|----------|----------|
| "Rate your Activities of Daily Living score" | "How independently can you do basic self-care?" |
| "Do you have hypertension?" | "Do you have high blood pressure?" |
| "Do you have cardiovascular disease?" | "Do you have heart disease or blood vessel problems?" |
| "What is your systolic blood pressure?" | "What is your blood pressure? (Top number)" |

### 3. **Zero Options Added**
All scale questions now properly include 0 as an option:
- Alcohol consumption: **0 = None at all**
- Physical activity: **0 = No exercise at all**
- Diet quality: **0 = Mostly junk food**
- Functional assessment: **0 = Cannot do these at all**
- Self-care: **0 = Need full help**

### 4. **Detailed Examples in Descriptions**

#### **Functional Assessment**
```
Examples: cooking, cleaning, shopping, managing money
0 = Cannot do these at all
10 = Do everything independently
```

#### **Self-Care (ADL)**
```
Examples: bathing, dressing, eating, using the toilet
0 = Need full help
10 = Completely independent
```

#### **BMI**
```
BMI = weight(kg) ÷ height(m)²
Normal range: 18.5-24.9
If unsure, enter 25
```

#### **Alcohol Consumption**
```
0 = None at all
5 = Moderate (1-2 drinks per day)
10 = Heavy (5+ drinks per day)
```

#### **Physical Activity**
```
0 = No exercise at all
5 = Moderate (walking 30 min/day)
10 = Very active (daily intense exercise)
```

#### **Diet Quality**
```
0 = Mostly junk food
5 = Balanced mix
10 = Very healthy (lots of fruits, vegetables, whole grains)
```

#### **Sleep Quality**
```
4 = Very poor (insomnia, frequent waking)
7 = Average
10 = Excellent (sleep through the night)
```

#### **Blood Pressure**
```
Systolic (top): Normal is around 120
Diastolic (bottom): Normal is around 80
If you don't know, enter 120/80
```

#### **Cardiovascular Disease**
```
Includes heart disease, stroke, or poor blood circulation
```

#### **Family History**
```
Family includes parents, siblings, grandparents
```

#### **High Blood Pressure**
```
High blood pressure (hypertension) is when your blood pressure 
is consistently above 140/90
Options: No, Yes, Not sure
```

### 5. **"Not Sure" Option**
Added "Not sure" option for hypertension question for users who don't know their blood pressure status.

### 6. **Default Value Guidance**
For questions that users might not know:
- BMI: "If unsure, enter 25"
- Blood pressure: "If you don't know, enter 120" / "enter 80"

## 🎨 Visual Design

### Info Box Styling
- **Background**: Light blue (`bg-blue-50`)
- **Border**: Blue (`border-blue-100`)
- **Icon**: ℹ️ emoji in blue
- **Text**: Clear, readable gray (`text-gray-600`)
- **Padding**: Comfortable spacing

### Example Display
```
┌─────────────────────────────────────────────────┐
│ Medical                                         │
│                                                 │
│ How much alcohol do you drink?                  │
│                                                 │
│ ┌─────────────────────────────────────────────┐ │
│ │ ℹ️ 0 = None at all, 5 = Moderate (1-2     │ │
│ │    drinks per day), 10 = Heavy (5+ drinks  │ │
│ │    per day)                                 │ │
│ └─────────────────────────────────────────────┘ │
│                                                 │
│ [0] [1] [2] [3] [4] [5] [6] [7] [8] [9] [10]  │
└─────────────────────────────────────────────────┘
```

## 🔧 Technical Implementation

### Frontend Changes
- Added `description?: string` field to Question interface
- Descriptions rendered in blue info box below question text
- Conditional rendering (only shows if description exists)

### Backend Changes
- Added handling for "Not sure" option in hypertension
- "Not sure" converts to 0.5 (middle ground between 0 and 1)

### Data Processing
```typescript
// Hypertension with "Not sure" option
if (rawAnswers.hypertension === 'Yes') {
  cognitiveObj.hypertension = 1;
} else if (rawAnswers.hypertension === 'Not sure') {
  cognitiveObj.hypertension = 0.5;  // Treat as uncertain
} else {
  cognitiveObj.hypertension = 0;
}
```

## ✅ Benefits

1. **Clearer Questions**: Medical terms explained in plain language
2. **Better UX**: Users understand what each scale value means
3. **More Accurate Data**: Users can select "0" when applicable
4. **Reduced Confusion**: Examples help users know what to report
5. **Accessibility**: "Not sure" option for uncertain cases
6. **Professional Look**: Consistent info boxes with helpful guidance

## 📊 All Questions with Descriptions (11 total)

1. ✅ Functional assessment (daily activities examples)
2. ✅ Self-care/ADL (bathing, dressing examples)
3. ✅ BMI (calculation formula + normal range)
4. ✅ Alcohol consumption (scale explanation)
5. ✅ Physical activity (exercise level examples)
6. ✅ Diet quality (food quality examples)
7. ✅ Sleep quality (sleep pattern examples)
8. ✅ Family history (family definition)
9. ✅ Cardiovascular disease (what it includes)
10. ✅ High blood pressure (hypertension definition)
11. ✅ Blood pressure numbers (systolic/diastolic explanation)

## 🎯 User-Friendly Features

### Clear Language
- ❌ "Hypertension" → ✅ "High blood pressure"
- ❌ "Systolic BP" → ✅ "Blood pressure (Top number)"
- ❌ "ADL score" → ✅ "Basic self-care"

### Practical Examples
- Cooking, cleaning, shopping (functional assessment)
- Bathing, dressing, eating (self-care)
- Walking 30 min/day (moderate activity)
- Fruits, vegetables, whole grains (healthy diet)

### Guidance for Unknowns
- BMI: Enter 25 if unsure
- Blood pressure: Enter 120/80 if unsure
- Hypertension: Select "Not sure" if unknown

## 🚀 Ready to Test

All improvements are complete and ready for user testing. The cognitive assessment is now:
- ✅ More user-friendly
- ✅ Less confusing
- ✅ Better documented
- ✅ More accessible
- ✅ Professionally presented
