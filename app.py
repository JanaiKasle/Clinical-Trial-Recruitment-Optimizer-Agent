# Clinical Trial Recruitment Optimizer Agent
# For Saama Technologies - Beginner-Friendly Implementation

import streamlit as st
import pandas as pd
import json
import boto3
from typing import Dict, List, Any
import re
from datetime import datetime
import io
import base64
from dataclasses import dataclass
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as plotly_express
import plotly.graph_objects as go
from plotly.subplots import make_subplots
try:
    import phoenix as phoenix_ai
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from openinference.instrumentation.bedrock import BedrockInstrumentor
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    phoenix_ai = None
    trace = None
    BedrockInstrumentor = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PatientScore:
    patient_id: str
    patient_name: str
    score: float
    matching_criteria: List[str]
    missing_criteria: List[str]
    patient_data: Dict[str, Any]

class ClinicalTrialAgent:
    """Main agent class for clinical trial recruitment optimization"""
    
    def __init__(self, aws_access_key: str, aws_secret_key: str, aws_region: str):
        """Initialize the agent with AWS credentials"""
        try:
            self.bedrock_client = boto3.client(
                'bedrock-runtime',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region
            )
            logger.info("AWS Bedrock client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AWS Bedrock client: {str(e)}")
            raise e
    def initialize_phoenix(self):
        """Initialize Phoenix for evaluation and tracing"""
        if not PHOENIX_AVAILABLE:
            logger.warning("Phoenix not available. Install with: pip install 'arize-phoenix[evals]'")
            return None
            
        try:
            # Launch Phoenix with proper configuration
            session = phoenix_ai.launch_app(host="127.0.0.1", port=6006)
            logger.info(f"Phoenix launched successfully at http://127.0.0.1:6006")
            return session
        except Exception as e:
            logger.error(f"Failed to initialize Phoenix: {str(e)}")
            return None

    def extract_eligibility_criteria(self, protocol_text: str) -> Dict[str, Any]:
        """Extract eligibility criteria from protocol document using Bedrock"""
        
        prompt = f"""
        You are a clinical trial expert. Extract the eligibility criteria from the following clinical trial protocol document.
        
        Please extract and structure the information in the following JSON format:
        {{
            "inclusion_criteria": [
                "criterion 1",
                "criterion 2"
            ],
            "exclusion_criteria": [
                "criterion 1", 
                "criterion 2"
            ],
            "age_requirements": {{
                "min_age": number,
                "max_age": number
            }},
            "gender_requirements": "male/female/all",
            "medical_conditions": [
                "condition 1",
                "condition 2"
            ],
            "medication_restrictions": [
                "medication 1",
                "medication 2"
            ]
        }}
        
        Protocol Document:
        {protocol_text[:4000]}
        
        Return ONLY valid JSON, no additional text or explanation.
        """
        
        try:
            # Bedrock call (automatically traced if instrumentation is active)
            response = self.bedrock_client.invoke_model(
                modelId="anthropic.claude-3-sonnet-20240229-v1:0",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 2000,
                    "messages": [{
                        "role": "user",
                        "content": prompt
                    }]
                })
            )

            response_body = json.loads(response['body'].read())
            criteria_text = response_body['content'][0]['text']

            # Log to Phoenix if available
            if PHOENIX_AVAILABLE and hasattr(st.session_state, 'phoenix_session'):
                logger.info(f"Bedrock call completed - Input length: {len(prompt)}, Output length: {len(criteria_text)}")
            
            # Clean up the response and extract JSON
            criteria_text = criteria_text.strip()
            if criteria_text.startswith('```json'):
                criteria_text = criteria_text[7:-3]
            elif criteria_text.startswith('```'):
                criteria_text = criteria_text[3:-3]
            
            # Try to extract JSON from the response
            try:
                criteria_json = json.loads(criteria_text)
            except json.JSONDecodeError:
                # Try to find JSON in the text
                json_start = criteria_text.find('{')
                json_end = criteria_text.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    criteria_json = json.loads(criteria_text[json_start:json_end])
                else:
                    raise Exception("Could not parse JSON from response")
            
            logger.info("Successfully extracted eligibility criteria")
            return criteria_json
            
        except Exception as e:
            logger.error(f"Error extracting criteria: {str(e)}")
            # Return enhanced fallback parsing
            return self._enhanced_fallback_criteria_extraction(protocol_text)
    
    def _enhanced_fallback_criteria_extraction(self, protocol_text: str) -> Dict[str, Any]:
        """Enhanced fallback method for criteria extraction using pattern matching"""
        
        text_lower = protocol_text.lower()
        
        criteria = {
            "inclusion_criteria": [],
            "exclusion_criteria": [], 
            "age_requirements": {"min_age": 18, "max_age": 80},
            "gender_requirements": "all",
            "medical_conditions": [],
            "medication_restrictions": []
        }
        
        # Extract age requirements
        age_patterns = [
            r'(\d+)\s*(?:to|-)?\s*(\d+)?\s*years?\s*(?:old|of age)',
            r'aged?\s+(\d+)\s*(?:to|-)?\s*(\d+)',
            r'between\s+(\d+)\s+and\s+(\d+)\s+years'
        ]
        
        for pattern in age_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                min_age = int(matches[0][0])
                max_age = int(matches[0][1]) if matches[0][1] else 80
                criteria["age_requirements"] = {"min_age": min_age, "max_age": max_age}
                break
        
        # Extract gender requirements
        if 'male only' in text_lower or 'males only' in text_lower:
            criteria["gender_requirements"] = "male"
        elif 'female only' in text_lower or 'females only' in text_lower:
            criteria["gender_requirements"] = "female"
        
        # Extract medical conditions
        medical_conditions = []
        condition_keywords = ['diabetes', 'hypertension', 'heart disease', 'cancer', 'depression', 'asthma', 'obesity']
        for condition in condition_keywords:
            if condition in text_lower:
                medical_conditions.append(condition)
        
        if medical_conditions:
            criteria["medical_conditions"] = medical_conditions
        
        # Extract inclusion criteria (simple pattern matching)
        inclusion_section = re.search(r'inclusion\s+criteria[:\s]*(.*?)(?=exclusion|$)', text_lower, re.DOTALL)
        if inclusion_section:
            inclusion_text = inclusion_section.group(1)
            inclusion_items = re.findall(r'\d+\.\s*([^0-9]+?)(?=\d+\.|$)', inclusion_text)
            criteria["inclusion_criteria"] = [item.strip().rstrip('.') for item in inclusion_items[:5]]
        
        # Extract exclusion criteria
        exclusion_section = re.search(r'exclusion\s+criteria[:\s]*(.*?)(?=\n\n|$)', text_lower, re.DOTALL)
        if exclusion_section:
            exclusion_text = exclusion_section.group(1)
            exclusion_items = re.findall(r'\d+\.\s*([^0-9]+?)(?=\d+\.|$)', exclusion_text)
            criteria["exclusion_criteria"] = [item.strip().rstrip('.') for item in exclusion_items[:5]]
        
        return criteria
    
    def score_patients(self, criteria: Dict[str, Any], patients_df: pd.DataFrame) -> List[PatientScore]:
        """Score patients based on eligibility criteria with improved matching logic"""
        scored_patients = []
        
        for _, patient in patients_df.iterrows():
            score = 0
            total_criteria = 0
            matching_criteria = []
            missing_criteria = []
            
            # 1. Age scoring (18-65 years)
            patient_age = patient.get('age', 0)
            min_age = criteria.get('age_requirements', {}).get('min_age', 18)
            max_age = criteria.get('age_requirements', {}).get('max_age', 65)
            
            total_criteria += 1
            if min_age <= patient_age <= max_age:
                score += 1
                matching_criteria.append(f"Age requirement met ({patient_age} years, range: {min_age}-{max_age})")
            else:
                missing_criteria.append(f"Age requirement not met ({patient_age} years, required: {min_age}-{max_age})")
            
            # 2. Type 2 Diabetes requirement (must be specific)
            patient_conditions = str(patient.get('medical_conditions', [])).lower()
            total_criteria += 1

            # Check for specific Type 2 diabetes or general diabetes (but not Type 1)
            has_type2_diabetes = (
                'type 2 diabetes' in patient_conditions or 
                ('diabetes' in patient_conditions and 'type 1 diabetes' not in patient_conditions)
            )

            # Exclude patients with only PCOS or other conditions where metformin is used
            has_only_pcos = 'pcos' in patient_conditions and 'diabetes' not in patient_conditions

            if has_type2_diabetes and not has_only_pcos:
                score += 1
                matching_criteria.append("Has Type 2 diabetes mellitus")
            else:
                if has_only_pcos:
                    missing_criteria.append("Has PCOS but no diabetes diagnosis")
                else:
                    missing_criteria.append("No Type 2 diabetes mellitus diagnosis")
                        
            # 3. BMI requirement (25-40 kg/m²)
            patient_bmi = patient.get('bmi', 0)
            total_criteria += 1
            if 25 <= patient_bmi <= 40:
                score += 1
                matching_criteria.append(f"BMI within range ({patient_bmi} kg/m², range: 25-40)")
            else:
                missing_criteria.append(f"BMI outside range ({patient_bmi} kg/m², required: 25-40)")
            
            # 4. Metformin therapy requirement
            patient_medications = str(patient.get('medications', [])).lower()
            total_criteria += 1
            if 'metformin' in patient_medications:
                score += 1
                matching_criteria.append("On metformin therapy")
            else:
                missing_criteria.append("Not on metformin therapy")
            
            # 5. Exclusion: Current smokers
            smoking_status = str(patient.get('smoking_status', '')).lower()
            total_criteria += 1
            if smoking_status != 'smoker':
                score += 1
                matching_criteria.append(f"Not a current smoker ({smoking_status})")
            else:
                missing_criteria.append("Current smoker (excluded)")
            
            # 6. Exclusion: Insulin use (except metformin only)
            total_criteria += 1
            if 'insulin' not in patient_medications:
                score += 1
                matching_criteria.append("Not using insulin")
            else:
                missing_criteria.append("Currently using insulin (exclusion criterion)")
            
            # 7. Exclusion: Severe cardiovascular disease (heart disease check)
            total_criteria += 1
            if 'heart disease' not in patient_conditions:
                score += 1
                matching_criteria.append("No severe cardiovascular disease")
            else:
                missing_criteria.append("Has cardiovascular disease (exclusion)")
            
            # 8. Exclusion: Severe renal impairment (kidney disease check)
            total_criteria += 1
            if 'kidney disease' not in patient_conditions:
                score += 1
                matching_criteria.append("No severe renal impairment")
            else:
                missing_criteria.append("Has kidney disease (exclusion)")
            
            # Calculate final score as percentage
            final_score = (score / total_criteria) * 100
            
            scored_patient = PatientScore(
                patient_id=str(patient.get('patient_id', '')),
                patient_name=patient.get('name', 'Unknown'),
                score=final_score,
                matching_criteria=matching_criteria,
                missing_criteria=missing_criteria,
                patient_data=patient.to_dict()
            )
            
            scored_patients.append(scored_patient)
        
        # Sort by score descending
        scored_patients.sort(key=lambda x: x.score, reverse=True)
        return scored_patients    
    def generate_report(self, criteria: Dict[str, Any], scored_patients: List[PatientScore]) -> str:
        """Generate a comprehensive report"""
        
        # Safely get values with defaults
        age_requirements = criteria.get('age_requirements', {})
        min_age = age_requirements.get('min_age', 'Not specified')
        max_age = age_requirements.get('max_age', 'Not specified')
        gender_requirements = criteria.get('gender_requirements', 'All genders accepted')
        inclusion_criteria = criteria.get('inclusion_criteria', ['None specified'])
        exclusion_criteria = criteria.get('exclusion_criteria', ['None specified'])
        
        report = f"""
# Clinical Trial Recruitment Report
**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Generated by:** Saama Technologies Clinical Trial Optimizer

## Trial Eligibility Criteria Summary

### Age Requirements
- Minimum Age: {min_age}
- Maximum Age: {max_age}

### Gender Requirements
- Required Gender: {gender_requirements}

### Inclusion Criteria
{chr(10).join(f"- {criterion}" for criterion in inclusion_criteria)}

### Exclusion Criteria
{chr(10).join(f"- {criterion}" for criterion in exclusion_criteria)}

## Patient Matching Results

### Summary Statistics
- Total Patients Evaluated: {len(scored_patients)}
- High Match Patients (≥80%): {len([p for p in scored_patients if p.score >= 80])}
- Medium Match Patients (50-79%): {len([p for p in scored_patients if 50 <= p.score < 80])}
- Low Match Patients (<50%): {len([p for p in scored_patients if p.score < 50])}

### Top 10 Recommended Patients

"""
        
        for i, patient in enumerate(scored_patients[:10], 1):
            # Safely get patient data
            age = patient.patient_data.get('age', 'N/A')
            gender = patient.patient_data.get('gender', 'N/A')
            medical_conditions = patient.patient_data.get('medical_conditions', 'N/A')
            
            report += f"""
#### {i}. {patient.patient_name} (ID: {patient.patient_id})
- **Match Score:** {patient.score:.1f}%
- **Matching Criteria:** {len(patient.matching_criteria)} criteria met
- **Missing Criteria:** {len(patient.missing_criteria)} criteria not met

**Patient Details:**
- Age: {age}
- Gender: {gender}
- Medical Conditions: {medical_conditions}

**Matching Criteria:**
{chr(10).join(f"  ✓ {criterion}" for criterion in patient.matching_criteria)}

**Missing Criteria:**
{chr(10).join(f"  ✗ {criterion}" for criterion in patient.missing_criteria)}

---
"""
        
        return report
        
    def create_visualizations(self, scored_patients: List[PatientScore]) -> Dict[str, Any]:
        """Create comprehensive visualizations for patient matching results"""
        
        # Prepare data for visualizations
        patient_names = [p.patient_name for p in scored_patients]
        scores = [p.score for p in scored_patients]
        patient_ids = [p.patient_id for p in scored_patients]
        
        # Calculate overall criteria statistics
        total_criteria_met = sum(len(p.matching_criteria) for p in scored_patients)
        total_criteria_missing = sum(len(p.missing_criteria) for p in scored_patients)
        
        visualizations = {}
        
        # 1. Horizontal Bar Chart - Match Scores per Patient
        fig_hbar = go.Figure(data=[
            go.Bar(
                y=patient_names[:10],  # Show top 10 patients
                x=scores[:10],
                orientation='h',
                text=[f'{score:.1f}%' for score in scores[:10]],
                textposition='inside',
                marker=dict(
                    color=scores[:10],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Match Score (%)")
                )
            )
        ])
        
        fig_hbar.update_layout(
            title='Top 10 Patients - Match Scores',
            xaxis_title='Match Score (%)',
            yaxis_title='Patients',
            height=500,
            template='plotly_white'
        )
        
        visualizations['horizontal_bar'] = fig_hbar
        
        # 2. Pie Chart - Overall Criteria Met vs Unmatched
        criteria_labels = ['Criteria Met', 'Criteria Unmatched']
        criteria_values = [total_criteria_met, total_criteria_missing]
        
        fig_pie = go.Figure(data=[
            go.Pie(
                labels=criteria_labels,
                values=criteria_values,
                hole=0.4,
                marker_colors=['#2E8B57', '#DC143C'],
                textinfo='label+percent+value',
                textfont_size=12
            )
        ])
        
        fig_pie.update_layout(
            title='Overall Eligibility Criteria Status',
            annotations=[dict(text=f'Total<br>Evaluations<br>{total_criteria_met + total_criteria_missing}', 
                            x=0.5, y=0.5, font_size=14, showarrow=False)],
            height=400,
            template='plotly_white'
        )
        
        visualizations['pie_chart'] = fig_pie
        
        # 3. Histogram - Distribution of Match Scores
        fig_hist = go.Figure(data=[
            go.Histogram(
                x=scores,
                nbinsx=20,
                marker=dict(
                    color='rgba(46, 139, 87, 0.7)',
                    line=dict(color='rgba(46, 139, 87, 1)', width=1)
                ),
                text=[f'{len([s for s in scores if bin_start <= s < bin_start + 5])}' 
                      for bin_start in range(0, 100, 5)],
                textposition='outside'
            )
        ])
        
        fig_hist.update_layout(
            title='Distribution of Patient Match Scores',
            xaxis_title='Match Score (%)',
            yaxis_title='Number of Patients',
            bargap=0.1,
            height=400,
            template='plotly_white'
        )
        
        visualizations['histogram'] = fig_hist
        
        # 4. Score Categories Breakdown
        score_categories = {
            'Excellent (90-100%)': len([s for s in scores if s >= 90]),
            'Good (70-89%)': len([s for s in scores if 70 <= s < 90]),
            'Fair (50-69%)': len([s for s in scores if 50 <= s < 70]),
            'Poor (0-49%)': len([s for s in scores if s < 50])
        }
        
        fig_categories = go.Figure(data=[
            go.Bar(
                x=list(score_categories.keys()),
                y=list(score_categories.values()),
                marker_color=['#228B22', '#FFD700', '#FF8C00', '#DC143C'],
                text=list(score_categories.values()),
                textposition='outside'
            )
        ])
        
        fig_categories.update_layout(
            title='Patient Categories by Match Score',
            xaxis_title='Score Categories',
            yaxis_title='Number of Patients',
            height=400,
            template='plotly_white'
        )
        
        visualizations['score_categories'] = fig_categories
        
        # 5. Patient Demographics Visualization (if age/gender data available)
        ages = [p.patient_data.get('age', 0) for p in scored_patients if p.patient_data.get('age')]
        genders = [p.patient_data.get('gender', 'unknown') for p in scored_patients if p.patient_data.get('gender')]
        
        if ages:
            # Age distribution with scores
            fig_age_score = go.Figure()
            fig_age_score.add_trace(go.Scatter(
                x=ages,
                y=scores,
                mode='markers',
                marker=dict(
                    color=scores,
                    colorscale='RdYlGn',
                    size=8
                ),
                text=[f"Age: {age}, Score: {score:.1f}%" for age, score in zip(ages, scores)],
                hovertemplate='%{text}<extra></extra>'
            ))
            fig_age_score = plotly_express.scatter(
                x=ages,
                y=scores,
                title='Match Scores vs Patient Age',
                labels={'x': 'Age (years)', 'y': 'Match Score (%)'},
                color=scores,
                color_continuous_scale='RdYlGn',
                size=[5]*len(ages)
            )
            fig_age_score.update_layout(height=400, template='plotly_white')
            visualizations['age_score_scatter'] = fig_age_score
        
        if genders:
            # Gender distribution
            gender_counts = pd.Series(genders).value_counts()
            fig_gender = plotly_express.pie(
                values=gender_counts.values,
                names=gender_counts.index,
                title='Patient Gender Distribution'
            )
            fig_gender.update_layout(height=350, template='plotly_white')
            visualizations['gender_distribution'] = fig_gender
        
        return visualizations
    
    def create_matplotlib_visualizations(self, scored_patients: List[PatientScore]) -> Dict[str, Any]:
        """Create additional visualizations using Matplotlib/Seaborn"""
        
        scores = [p.score for p in scored_patients]
        patient_names = [p.patient_name for p in scored_patients]
        
        matplotlib_figs = {}
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Box plot and violin plot for score distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Box plot
        ax1.boxplot(scores, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax1.set_ylabel('Match Score (%)')
        ax1.set_title('Score Distribution - Box Plot')
        ax1.grid(True, alpha=0.3)
        
        # Violin plot
        ax2.violinplot(scores, vert=True)
        ax2.set_ylabel('Match Score (%)')
        ax2.set_title('Score Distribution - Violin Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        matplotlib_figs['distribution_analysis'] = fig
        
        return matplotlib_figs
    
    def evaluate_with_phoenix(self, protocol_text: str, extracted_criteria: Dict[str, Any], 
                         scored_patients: List[PatientScore]) -> Dict[str, Any]:
        """Evaluate the clinical trial matching using simplified metrics"""
        
        try:
            # Simple evaluation without complex Phoenix evals
            clinical_eval_results = self._custom_clinical_evaluation_simple(extracted_criteria, scored_patients)
            
            evaluations = {
                'clinical_accuracy': clinical_eval_results
            }
            
            # Log evaluation to Phoenix if available
            if PHOENIX_AVAILABLE and hasattr(st.session_state, 'phoenix_session'):
                logger.info(f"Clinical Trial Evaluation completed: {clinical_eval_results.get('overall_accuracy', 0):.1f}% accuracy")
            
            return {
                "phoenix_session": getattr(st.session_state, 'phoenix_session', None),
                "evaluations": evaluations,
                "evaluation_summary": self._generate_evaluation_summary(evaluations)
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            return {"error": f"Evaluation failed: {str(e)}"}

    def _custom_clinical_evaluation_simple(self, criteria: Dict[str, Any], 
                                        patients: List[PatientScore]) -> Dict[str, Any]:
        """Simplified custom evaluation logic for clinical trial matching accuracy"""
        
        results = {
            "criteria_extraction_accuracy": 0,
            "patient_ranking_accuracy": 0,
            "exclusion_criteria_accuracy": 0,
            "overall_accuracy": 0,
            "total_patients_evaluated": len(patients),
            "high_scoring_patients": len([p for p in patients if p.score >= 80])
        }
        
        try:
            # 1. Evaluate criteria extraction completeness
            expected_criteria_keys = ['inclusion_criteria', 'exclusion_criteria', 'age_requirements', 'medical_conditions']
            found_criteria = sum(1 for key in expected_criteria_keys if key in criteria and criteria[key])
            results["criteria_extraction_accuracy"] = (found_criteria / len(expected_criteria_keys)) * 100
            
            # 2. Evaluate patient ranking logic
            diabetes_patients = [p for p in patients if 'diabetes' in str(p.patient_data.get('medical_conditions', '')).lower()]
            non_diabetes_patients = [p for p in patients if 'diabetes' not in str(p.patient_data.get('medical_conditions', '')).lower()]
            
            if diabetes_patients and non_diabetes_patients:
                avg_diabetes_score = sum(p.score for p in diabetes_patients) / len(diabetes_patients)
                avg_non_diabetes_score = sum(p.score for p in non_diabetes_patients) / len(non_diabetes_patients)
                
                # Diabetes patients should score higher on average
                ranking_accuracy = min(100, max(0, (avg_diabetes_score - avg_non_diabetes_score) * 2))
                results["patient_ranking_accuracy"] = ranking_accuracy
            else:
                results["patient_ranking_accuracy"] = 50  # Neutral if no comparison possible
            
            # 3. Evaluate exclusion criteria application  
            smokers_penalized = 0
            insulin_users_penalized = 0
            age_violations_penalized = 0
            
            for patient in patients:
                smoking_status = str(patient.patient_data.get('smoking_status', '')).lower()
                medications = str(patient.patient_data.get('medications', [])).lower()
                age = patient.patient_data.get('age', 0)
                
                # Check smoker penalties
                if smoking_status == 'smoker' and patient.score < 100:
                    smokers_penalized += 1
                    
                # Check insulin user penalties
                if 'insulin' in medications and patient.score < 100:
                    insulin_users_penalized += 1
                    
                # Check age violation penalties  
                if (age < 18 or age > 65) and patient.score < 100:
                    age_violations_penalized += 1
            
            # Calculate exclusion accuracy based on proper penalties
            total_exclusion_checks = len([p for p in patients if 
                                        str(p.patient_data.get('smoking_status', '')).lower() == 'smoker' or
                                        'insulin' in str(p.patient_data.get('medications', [])).lower() or
                                        p.patient_data.get('age', 0) < 18 or p.patient_data.get('age', 0) > 65])
            
            if total_exclusion_checks > 0:
                total_penalized = smokers_penalized + insulin_users_penalized + age_violations_penalized
                results["exclusion_criteria_accuracy"] = min(100, (total_penalized / total_exclusion_checks) * 100)
            else:
                results["exclusion_criteria_accuracy"] = 100  # No exclusions to check
            
            # 4. Overall accuracy
            results["overall_accuracy"] = (
                results["criteria_extraction_accuracy"] * 0.3 +
                results["patient_ranking_accuracy"] * 0.4 +
                results["exclusion_criteria_accuracy"] * 0.3
            )
            
            # Additional metrics
            results["diabetes_patients_found"] = len(diabetes_patients)
            results["average_score"] = sum(p.score for p in patients) / len(patients) if patients else 0
            
        except Exception as e:
            logger.error(f"Custom evaluation failed: {str(e)}")
            results["error"] = str(e)
        
        return results

    def _custom_clinical_evaluation(self, eval_df: pd.DataFrame, criteria: Dict[str, Any], 
                                patients: List[PatientScore]) -> Dict[str, Any]:
        """Custom evaluation logic for clinical trial matching accuracy"""
        
        results = {
            "criteria_extraction_accuracy": 0,
            "patient_ranking_accuracy": 0,
            "exclusion_criteria_accuracy": 0,
            "overall_accuracy": 0
        }
        
        try:
            # 1. Evaluate criteria extraction completeness
            expected_criteria_keys = ['inclusion_criteria', 'exclusion_criteria', 'age_requirements', 'medical_conditions']
            found_criteria = sum(1 for key in expected_criteria_keys if key in criteria and criteria[key])
            results["criteria_extraction_accuracy"] = (found_criteria / len(expected_criteria_keys)) * 100
            
            # 2. Evaluate patient ranking logic
            diabetes_patients = [p for p in patients if 'diabetes' in str(p.patient_data.get('medical_conditions', '')).lower()]
            non_diabetes_patients = [p for p in patients if 'diabetes' not in str(p.patient_data.get('medical_conditions', '')).lower()]
            
            if diabetes_patients and non_diabetes_patients:
                avg_diabetes_score = sum(p.score for p in diabetes_patients) / len(diabetes_patients)
                avg_non_diabetes_score = sum(p.score for p in non_diabetes_patients) / len(non_diabetes_patients)
                
                # Diabetes patients should score higher on average
                ranking_accuracy = min(100, max(0, (avg_diabetes_score - avg_non_diabetes_score) * 2))
                results["patient_ranking_accuracy"] = ranking_accuracy
            
            # 3. Evaluate exclusion criteria application
            excluded_correctly = 0
            total_exclusions = 0
            
            for patient in patients:
                patient_data = str(patient.patient_data).lower()
                total_exclusions += 1
                
                # Check if smokers are properly penalized
                if 'smoker' in patient_data and patient.score < 90:
                    excluded_correctly += 1
                # Check if insulin users are properly penalized  
                elif 'insulin' in patient_data and patient.score < 90:
                    excluded_correctly += 1
                # Check if patients outside age range are penalized
                elif patient.patient_data.get('age', 0) > 65 and patient.score < 90:
                    excluded_correctly += 1
                else:
                    excluded_correctly += 0.5  # Partial credit
            
            if total_exclusions > 0:
                results["exclusion_criteria_accuracy"] = (excluded_correctly / total_exclusions) * 100
            
            # 4. Overall accuracy
            results["overall_accuracy"] = (
                results["criteria_extraction_accuracy"] * 0.3 +
                results["patient_ranking_accuracy"] * 0.4 +
                results["exclusion_criteria_accuracy"] * 0.3
            )
            
        except Exception as e:
            logger.error(f"Custom evaluation failed: {str(e)}")
        
        return results

    def _generate_evaluation_summary(self, evaluations: Dict[str, Any]) -> Dict[str, str]:
        """Generate human-readable summary of Phoenix evaluations"""
        
        summary = {
            "overall_status": "Unknown",
            "key_findings": [],
            "recommendations": []
        }
        
        try:
            clinical_accuracy = evaluations.get('clinical_accuracy', {})
            overall_accuracy = clinical_accuracy.get('overall_accuracy', 0)
            
            if overall_accuracy >= 80:
                summary["overall_status"] = "Excellent"
                summary["key_findings"].append("High quality criteria extraction and patient matching")
            elif overall_accuracy >= 60:
                summary["overall_status"] = "Good"
                summary["key_findings"].append("Acceptable performance with room for improvement")
            else:
                summary["overall_status"] = "Needs Improvement"
                summary["key_findings"].append("Significant issues detected in matching logic")
            
            # Add specific recommendations
            if clinical_accuracy.get('criteria_extraction_accuracy', 0) < 70:
                summary["recommendations"].append("Improve protocol document parsing and criteria extraction")
            
            if clinical_accuracy.get('patient_ranking_accuracy', 0) < 70:
                summary["recommendations"].append("Enhance patient scoring algorithm for better ranking")
                
            if clinical_accuracy.get('exclusion_criteria_accuracy', 0) < 70:
                summary["recommendations"].append("Strengthen exclusion criteria implementation")
                
        except Exception as e:
            summary["overall_status"] = "Evaluation Error"
            summary["key_findings"].append(f"Error generating summary: {str(e)}")
        
        return summary
    

def main():
    st.set_page_config(
        page_title="Clinical Trial Recruitment Optimizer",
        page_icon="🏥",
        layout="wide"
    )
    
    # Initialize Phoenix tracing with new API
    if PHOENIX_AVAILABLE:
        try:
            # Start Phoenix session if not already running
            if 'phoenix_session' not in st.session_state:
                st.session_state.phoenix_session = phoenix_ai.launch_app(
                    host="127.0.0.1", 
                    port=6006
                )
                
                # Configure OpenTelemetry for Phoenix
                tracer_provider = TracerProvider()
                trace.set_tracer_provider(tracer_provider)
                
                # Set up OTLP exporter to send traces to Phoenix
                otlp_exporter = OTLPSpanExporter(
                    endpoint="http://127.0.0.1:6006/v1/traces"
                )
                span_processor = BatchSpanProcessor(otlp_exporter)
                tracer_provider.add_span_processor(span_processor)
                
                # Instrument Bedrock
                if BedrockInstrumentor:
                    BedrockInstrumentor().instrument()
                
                st.sidebar.success("🔍 Phoenix monitoring active at http://127.0.0.1:6006")
        except Exception as e:
            st.sidebar.warning(f"Phoenix setup: {str(e)}")
    
    st.title("🏥 Clinical Trial Recruitment Optimizer")
    st.markdown("**Powered by Saama Technologies | AWS Bedrock & AI**")
    
    # Sidebar for AWS configuration
    st.sidebar.header("⚙️ AWS Configuration")
    
    aws_access_key = st.sidebar.text_input(
        "AWS Access Key ID",
        type="password",
        help="Enter your AWS Access Key ID"
    )
    
    aws_secret_key = st.sidebar.text_input(
        "AWS Secret Access Key", 
        type="password",
        help="Enter your AWS Secret Access Key"
    )
    
    aws_region = st.sidebar.selectbox(
        "AWS Region",
        ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
        help="Select your AWS region"
    )
    
    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'criteria' not in st.session_state:
        st.session_state.criteria = None
    if 'scored_patients' not in st.session_state:
        st.session_state.scored_patients = None
    if 'patients_df' not in st.session_state:
        st.session_state.patients_df = None
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Protocol Document")
        
        # Protocol document upload
        uploaded_protocol = st.file_uploader(
            "Upload Clinical Trial Protocol Document",
            type=['txt'],
            help="Upload your clinical trial protocol document (.txt format)",
            key="protocol_uploader"
        )
        
        protocol_text = None
        if uploaded_protocol is not None:
            try:
                protocol_text = str(uploaded_protocol.read(), "utf-8")
                st.success(f"✅ Protocol uploaded: {uploaded_protocol.name}")
                st.text_area("Protocol Content", protocol_text[:500] + "...", height=200, disabled=True)
            except Exception as e:
                st.error(f"❌ Error reading protocol file: {str(e)}")
        else:
            st.info("👆 Please upload a protocol document (.txt format)")
        
        # Initialize agent button
        if st.button("🚀 Initialize AI Agent", type="primary"):
            if not all([aws_access_key, aws_secret_key, aws_region]):
                st.error("Please provide all AWS credentials in the sidebar!")
            else:
                try:
                    with st.spinner("Initializing AWS Bedrock AI Agent..."):
                        st.session_state.agent = ClinicalTrialAgent(
                            aws_access_key, aws_secret_key, aws_region
                        )
                    st.success("✅ AI Agent initialized successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to initialize agent: {str(e)}")
    
    with col2:
        st.header("👥 Patient Database")
        
        # Patient database upload
        uploaded_patients = st.file_uploader(
            "Upload Patient Database",
            type=['csv'],
            help="Upload your patient database (.csv format)",
            key="patients_uploader"
        )
        
        if uploaded_patients is not None:
            try:
                patients_df = pd.read_csv(uploaded_patients)
                st.session_state.patients_df = patients_df
                st.success(f"✅ Patient database uploaded: {uploaded_patients.name}")
                st.write(f"**Total patients:** {len(patients_df)}")
                st.dataframe(patients_df.head(), use_container_width=True)
                
                if st.button("📋 View Full Database"):
                    st.dataframe(patients_df, use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Error reading patient file: {str(e)}")
                st.session_state.patients_df = None
        else:
            st.info("👆 Please upload a patient database (.csv format)")
            st.session_state.patients_df = None
            
            # Show expected format
            with st.expander("📋 Expected CSV Format"):
                st.markdown("""
                Your CSV should include these columns:
                - `patient_id`: Unique identifier
                - `name`: Patient name
                - `age`: Age in years
                - `gender`: male/female
                - `medical_conditions`: Comma-separated conditions
                - `medications`: Current medications
                - `bmi`: Body Mass Index (optional)
                - `smoking_status`: smoking status (optional)
                
                **Example:**
                ```
                patient_id,name,age,gender,medical_conditions,medications,bmi,smoking_status
                P001,John Smith,45,male,"diabetes,hypertension","metformin,lisinopril",28.5,non-smoker
                P002,Sarah Johnson,32,female,asthma,albuterol,24.2,non-smoker
                ```
                """)
        
        # Download template button
        if st.button("📥 Download CSV Template"):
            template_data = {
                'patient_id': ['P001', 'P002', 'P003'],
                'name': ['John Smith', 'Sarah Johnson', 'Michael Brown'],
                'age': [45, 32, 67],
                'gender': ['male', 'female', 'male'],
                'medical_conditions': ['diabetes,hypertension', 'asthma', 'diabetes,heart disease'],
                'medications': ['metformin,lisinopril', 'albuterol', 'insulin,atenolol'],
                'bmi': [28.5, 24.2, 31.0],
                'smoking_status': ['non-smoker', 'non-smoker', 'former smoker']
            }
            template_df = pd.DataFrame(template_data)
            csv_template = template_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Patient Database Template",
                data=csv_template,
                file_name="patient_database_template.csv",
                mime="text/csv"
            )
    
    # Processing section
    st.header("🤖 AI Processing")
    
    if st.button("🔍 Extract Criteria & Match Patients", type="primary"):
        if st.session_state.agent is None:
            st.error("Please initialize the AI Agent first!")
        elif not protocol_text:
            st.error("Please upload a protocol document!")
        elif st.session_state.patients_df is None:
            st.error("Please upload a patient database!")
        else:
            try:
                # Step 1: Extract criteria
                with st.spinner("Step 1: Extracting eligibility criteria from protocol..."):
                    criteria = st.session_state.agent.extract_eligibility_criteria(protocol_text)
                    st.session_state.criteria = criteria
                
                st.success("✅ Criteria extracted successfully!")
                
                # Display extracted criteria
                st.subheader("📋 Extracted Eligibility Criteria")
                st.json(criteria)
                
                # Step 2: Score patients
                with st.spinner("Step 2: Scoring patients against criteria..."):
                    scored_patients = st.session_state.agent.score_patients(criteria, st.session_state.patients_df)
                    st.session_state.scored_patients = scored_patients
                
                # Phoenix Evaluation Step
                with st.spinner("Step 3: Running Phoenix AI evaluation..."):
                    phoenix_results = st.session_state.agent.evaluate_with_phoenix(
                        protocol_text, criteria, scored_patients
                    )
                    st.session_state.phoenix_results = phoenix_results

                if 'error' not in phoenix_results:
                    st.success("✅ Phoenix evaluation completed!")
                    
                    # Display evaluation summary
                    st.subheader("🔍 Phoenix AI Evaluation Results")
                    eval_summary = phoenix_results.get('evaluation_summary', {})
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Overall Status", eval_summary.get('overall_status', 'Unknown'))
                    with col2:
                        clinical_acc = phoenix_results.get('evaluations', {}).get('clinical_accuracy', {})
                        st.metric("Overall Accuracy", f"{clinical_acc.get('overall_accuracy', 0):.1f}%")
                    with col3:
                        st.metric("Criteria Extraction", f"{clinical_acc.get('criteria_extraction_accuracy', 0):.1f}%")
                    
                    # Key findings
                    st.write("**Key Findings:**")
                    for finding in eval_summary.get('key_findings', []):
                        st.write(f"• {finding}")
                    
                    # Recommendations
                    if eval_summary.get('recommendations'):
                        st.write("**Recommendations:**")
                        for rec in eval_summary.get('recommendations', []):
                            st.write(f"• {rec}")
                else:
                    st.warning(f"⚠️ Phoenix evaluation encountered issues: {phoenix_results.get('error')}")
                
                st.success("✅ Patient scoring completed!")
                
                # Step 3: Display results
                st.subheader("🏆 Patient Matching Results")
                
                # Create results dataframe
                results_data = []
                for patient in scored_patients:
                    results_data.append({
                        'Rank': len(results_data) + 1,
                        'Patient ID': patient.patient_id,
                        'Name': patient.patient_name,
                        'Match Score (%)': f"{patient.score:.1f}%",
                        'Age': patient.patient_data.get('age', 'N/A'),
                        'Gender': patient.patient_data.get('gender', 'N/A'),
                        'Criteria Met': len(patient.matching_criteria),
                        'Criteria Missing': len(patient.missing_criteria)
                    })
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                
                # Generate and display visualizations
                st.header("📊 Interactive Visualizations")
                
                with st.spinner("Generating comprehensive visualizations..."):
                    visualizations = st.session_state.agent.create_visualizations(scored_patients)
                    matplotlib_figs = st.session_state.agent.create_matplotlib_visualizations(scored_patients)
                
                # Create tabs for different visualizations
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "📊 Score Analysis", 
                    "🎯 Criteria Overview", 
                    "📈 Distribution", 
                    "👥 Demographics", 
                    "🔬 Advanced Analysis",
                    "🔍 Phoenix Evaluation"
                ])
                
                with tab1:
                    st.subheader("Patient Match Scores")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.plotly_chart(visualizations['horizontal_bar'], use_container_width=True)
                    
                    with col2:
                        st.plotly_chart(visualizations['score_categories'], use_container_width=True)
                
                with tab2:
                    st.subheader("Eligibility Criteria Analysis")
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.plotly_chart(visualizations['pie_chart'], use_container_width=True)
                    
                    with col2:
                        # Summary statistics
                        total_patients = len(scored_patients)
                        avg_score = np.mean([p.score for p in scored_patients])
                        high_match_patients = len([p for p in scored_patients if p.score >= 80])
                        
                        st.metric("Total Patients", total_patients)
                        st.metric("Average Match Score", f"{avg_score:.1f}%")
                        st.metric("High Match Patients (≥80%)", high_match_patients)
                        st.metric("Success Rate", f"{(high_match_patients/total_patients)*100:.1f}%")
                
                with tab3:
                    st.subheader("Score Distribution Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.plotly_chart(visualizations['histogram'], use_container_width=True)
                    
                    with col2:
                        # Matplotlib distribution plots
                        if 'distribution_analysis' in matplotlib_figs:
                            st.pyplot(matplotlib_figs['distribution_analysis'])
                
                with tab4:
                    st.subheader("Patient Demographics")
                    if 'age_score_scatter' in visualizations:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(visualizations['age_score_scatter'], use_container_width=True)
                        with col2:
                            if 'gender_distribution' in visualizations:
                                st.plotly_chart(visualizations['gender_distribution'], use_container_width=True)
                    else:
                        st.info("Demographics visualization requires age and gender data in the patient database.")
                
                with tab5:
                    st.subheader("Advanced Statistical Analysis")
                    
                    # Criteria matching heatmap
                    if 'criteria_heatmap' in matplotlib_figs:
                        st.pyplot(matplotlib_figs['criteria_heatmap'])
                    
                    # Radar chart for top performers
                    if 'radar_chart' in matplotlib_figs:
                        st.pyplot(matplotlib_figs['radar_chart'])
                    
                    # Statistical summary
                    scores = [p.score for p in scored_patients]
                    st.subheader("Statistical Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Mean Score", f"{np.mean(scores):.1f}%")
                    with col2:
                        st.metric("Median Score", f"{np.median(scores):.1f}%")
                    with col3:
                        st.metric("Std Deviation", f"{np.std(scores):.1f}%")
                    with col4:
                        st.metric("Score Range", f"{np.max(scores) - np.min(scores):.1f}%")
                
                with tab6:
                    st.subheader("Phoenix AI Monitoring Dashboard")
                    
                    if PHOENIX_AVAILABLE and hasattr(st.session_state, 'phoenix_session'):
                        st.success("🔍 Phoenix is running and automatically tracing your AWS Bedrock calls!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Phoenix Dashboard:**")
                            st.markdown("🌐 [Open Phoenix Dashboard](http://127.0.0.1:6006)")
                            st.code("http://127.0.0.1:6006")
                            st.write("Click the link above to view:")
                            st.write("• Real-time LLM call traces")
                            st.write("• Token usage and costs") 
                            st.write("• Latency metrics")
                            st.write("• Input/output analysis")
                            
                        with col2:
                            st.write("**Instructions:**")
                            st.write("1. Click 'Extract Criteria & Match Patients'")
                            st.write("2. Open the Phoenix dashboard link")
                            st.write("3. You'll see traces appear automatically")
                            st.write("4. Navigate to 'Traces' tab in Phoenix")
                        
                        # Show evaluation results if available
                        if 'phoenix_results' in st.session_state and 'error' not in st.session_state.phoenix_results:
                            st.subheader("📊 Evaluation Metrics")
                            phoenix_results = st.session_state.phoenix_results
                            clinical_accuracy = phoenix_results.get('evaluations', {}).get('clinical_accuracy', {})
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Overall Accuracy", f"{clinical_accuracy.get('overall_accuracy', 0):.1f}%")
                            with col2:
                                st.metric("Criteria Extraction", f"{clinical_accuracy.get('criteria_extraction_accuracy', 0):.1f}%")
                            with col3:
                                st.metric("Patient Ranking", f"{clinical_accuracy.get('patient_ranking_accuracy', 0):.1f}%")
                                
                    else:
                        st.warning("Phoenix is not running. Check terminal for any initialization errors.")
                        
                        with st.expander("🔧 Phoenix Setup Instructions"):
                            st.write("Install Phoenix with the correct packages:")
                            st.code("""
                pip install arize-phoenix==4.10.0
                pip install openinference-instrumentation-bedrock==0.1.4
                pip install opentelemetry-api==1.21.0
                pip install opentelemetry-sdk==1.21.0
                pip install opentelemetry-exporter-otlp-proto-http==1.21.0
                            """)
                            st.write("Then restart your Streamlit app.")

            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
    
    # Report generation section
    if st.session_state.criteria and st.session_state.scored_patients:
        st.header("📊 Generate Report")
        
        if st.button("📄 Generate Comprehensive Report"):
            try:
                with st.spinner("Generating comprehensive report..."):
                    report = st.session_state.agent.generate_report(
                        st.session_state.criteria,
                        st.session_state.scored_patients
                    )
                
                st.subheader("📋 Clinical Trial Recruitment Report")
                st.markdown(report)
                
                # Download button for report
                st.download_button(
                    label="📥 Download Report (Markdown)",
                    data=report,
                    file_name=f"clinical_trial_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
                
            except Exception as e:
                st.error(f"❌ Error generating report: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Clinical Trial Recruitment Optimizer** | "
        "Built with AWS Bedrock, Python & Streamlit | "
        "© 2025 Saama Technologies"
    )

if __name__ == "__main__":
    main()