import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
"""
Natural Language Clinical Report Generator
Generates human-readable diagnostic reports from ECG predictions and XAI outputs
"""

import numpy as np
import torch
import os
from datetime import datetime
import config
from src.models import ECGXAINet
from src.dataset import PTBXLDataset
from torch.utils.data import DataLoader


class ClinicalReportGenerator:
    """Generate natural language clinical reports from model predictions"""
    
    def __init__(self):
        self.disease_descriptions = {
            'NORM': {
                'full_name': 'Normal Sinus Rhythm',
                'severity': 'None',
                'description': 'No significant abnormalities detected',
                'recommendation': 'Continue routine monitoring'
            },
            'MI': {
                'full_name': 'Myocardial Infarction',
                'severity': 'Critical',
                'description': 'Evidence of heart muscle damage, possibly due to blocked coronary artery',
                'recommendation': 'URGENT: Immediate cardiology consultation required. Consider troponin levels and coronary angiography'
            },
            'STTC': {
                'full_name': 'ST/T Change',
                'severity': 'Moderate',
                'description': 'Abnormal ST segment or T wave patterns detected',
                'recommendation': 'Further evaluation recommended. Monitor for ischemic changes'
            },
            'CD': {
                'full_name': 'Conduction Disturbance',
                'severity': 'Moderate',
                'description': 'Abnormal electrical conduction through the heart',
                'recommendation': 'Consider electrophysiology study. Monitor for progression'
            },
            'HYP': {
                'full_name': 'Hypertrophy',
                'severity': 'Moderate',
                'description': 'Evidence of cardiac chamber enlargement',
                'recommendation': 'Echocardiography recommended. Evaluate for underlying causes'
            },
            'AFib': {
                'full_name': 'Atrial Fibrillation',
                'severity': 'High',
                'description': 'Irregular and often very rapid heart rhythm that can lead to blood clots in the heart',
                'recommendation': 'Assess stroke risk, consider anticoagulation and rate/rhythm control'
            }
        }
        
        self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    def _get_confidence_level(self, confidence):
        """Convert numeric confidence to descriptive text"""
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.75:
            return "High"
        elif confidence >= 0.6:
            return "Moderate"
        elif confidence >= 0.4:
            return "Low"
        else:
            return "Very Low"
    
    def _get_risk_assessment(self, predictions):
        """Generate risk assessment based on predictions"""
        max_pred = np.max(predictions)
        critical_diseases = ['MI', 'STTC']
        
        has_critical = any(predictions[i] > 0.5 for i, name in enumerate(config.DIAGNOSTIC_CLASSES) 
                          if name in critical_diseases)
        
        if has_critical and max_pred > 0.8:
            return "HIGH RISK - Immediate attention required"
        elif has_critical or max_pred > 0.7:
            return "MODERATE RISK - Prompt evaluation recommended"
        elif max_pred > 0.5:
            return "LOW RISK - Routine follow-up advised"
        else:
            return "MINIMAL RISK - Continue standard care"
    
    def _identify_key_leads(self, shap_values, top_n=3):
        """Identify most important leads from SHAP values"""
        if shap_values is None or len(shap_values.shape) < 2:
            return []
        
        # Calculate mean absolute SHAP value per lead
        lead_importance = np.abs(shap_values).mean(axis=1) if len(shap_values.shape) == 2 else np.abs(shap_values)
        
        # Get top N leads
        top_indices = np.argsort(lead_importance)[-top_n:][::-1]
        
        return [(self.lead_names[i], lead_importance[i]) for i in top_indices]
    
    def generate_report(self, predictions, true_labels=None, shap_values=None, sample_id=0):
        """
        Generate a comprehensive clinical report
        
        Args:
            predictions: Model prediction probabilities (5,)
            true_labels: Ground truth labels (5,) - optional
            shap_values: SHAP attribution values (12, 5000) - optional
            sample_id: Sample identifier
        
        Returns:
            str: Formatted clinical report
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Identify predicted diseases (threshold = 0.5)
        predicted_diseases = [(config.DIAGNOSTIC_CLASSES[i], predictions[i]) 
                             for i in range(len(predictions)) if predictions[i] > 0.5]
        
        # Sort by confidence
        predicted_diseases.sort(key=lambda x: x[1], reverse=True)
        
        # If no disease predicted above threshold, take the highest
        if not predicted_diseases:
            max_idx = np.argmax(predictions)
            predicted_diseases = [(config.DIAGNOSTIC_CLASSES[max_idx], predictions[max_idx])]
        
        # Generate report
        report = []
        report.append("=" * 80)
        report.append("ECG-XAI-NET AUTOMATED DIAGNOSTIC REPORT")
        report.append("=" * 80)
        report.append(f"Report Generated: {timestamp}")
        report.append(f"Sample ID: {sample_id}")
        report.append(f"Analysis Method: Hybrid CNN-Transformer with Explainable AI")
        report.append("")
        
        # PRIMARY FINDINGS
        report.append("-" * 80)
        report.append("PRIMARY FINDINGS")
        report.append("-" * 80)
        
        for i, (disease, confidence) in enumerate(predicted_diseases, 1):
            disease_info = self.disease_descriptions[disease]
            conf_level = self._get_confidence_level(confidence)
            
            report.append(f"\n{i}. {disease_info['full_name']} ({disease})")
            report.append(f"   Confidence: {confidence:.1%} ({conf_level})")
            report.append(f"   Severity: {disease_info['severity']}")
            report.append(f"   Description: {disease_info['description']}")
        
        # RISK ASSESSMENT
        report.append("\n" + "-" * 80)
        report.append("RISK ASSESSMENT")
        report.append("-" * 80)
        risk = self._get_risk_assessment(predictions)
        report.append(f"{risk}")
        
        # KEY FEATURES (if SHAP values available)
        if shap_values is not None:
            report.append("\n" + "-" * 80)
            report.append("KEY CONTRIBUTING FEATURES (AI Explanation)")
            report.append("-" * 80)
            
            key_leads = self._identify_key_leads(shap_values)
            if key_leads:
                report.append("Most influential ECG leads for this diagnosis:")
                for i, (lead, importance) in enumerate(key_leads, 1):
                    report.append(f"   {i}. Lead {lead} (Importance: {importance:.4f})")
                
                report.append("\nInterpretation: The model focused primarily on these leads when")
                report.append("making its diagnostic decision. Abnormalities in these leads are")
                report.append("most relevant to the predicted condition(s).")
        
        # RECOMMENDATIONS
        report.append("\n" + "-" * 80)
        report.append("CLINICAL RECOMMENDATIONS")
        report.append("-" * 80)
        
        for disease, confidence in predicted_diseases:
            if confidence > 0.5:
                disease_info = self.disease_descriptions[disease]
                report.append(f"• {disease_info['recommendation']}")
        
        # MODEL PERFORMANCE (if ground truth available)
        if true_labels is not None:
            report.append("\n" + "-" * 80)
            report.append("VALIDATION (Ground Truth Available)")
            report.append("-" * 80)
            
            true_diseases = [config.DIAGNOSTIC_CLASSES[i] for i in range(len(true_labels)) 
                           if true_labels[i] == 1]
            
            if true_diseases:
                report.append(f"Actual Diagnosis: {', '.join(true_diseases)}")
                
                # Check accuracy
                pred_set = set([d for d, c in predicted_diseases if c > 0.5])
                true_set = set(true_diseases)
                
                if pred_set == true_set:
                    report.append("Model Prediction: ✓ CORRECT")
                else:
                    report.append("Model Prediction: ✗ MISMATCH")
                    if pred_set - true_set:
                        report.append(f"   False Positives: {', '.join(pred_set - true_set)}")
                    if true_set - pred_set:
                        report.append(f"   Missed Diagnoses: {', '.join(true_set - pred_set)}")
        
        # DISCLAIMER
        report.append("\n" + "-" * 80)
        report.append("DISCLAIMER")
        report.append("-" * 80)
        report.append("This report is generated by an AI system for research purposes.")
        report.append("It should NOT be used as the sole basis for clinical decisions.")
        report.append("Always consult with qualified healthcare professionals.")
        report.append("=" * 80)
        
        return "\n".join(report)


def generate_GenAI_sample_reports(num_samples=3):
    """Generate clinical reports for sample ECG predictions"""
    
    print("\n" + "="*60)
    print("Generating Natural Language Clinical Reports...")
    print("="*60)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ECGXAINet(num_leads=config.NUM_LEADS, num_classes=config.NUM_CLASSES)
    
    model_path = os.path.join(config.BASE_DIR, 'models', 'best_model.pth')
    if not os.path.exists(model_path):
        print("Model not found. Skipping clinical report generation.")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    # Load test data
    test_dataset = PTBXLDataset(config.DATA_DIR, split='test', sampling_rate=config.SAMPLING_RATE)
    test_loader = DataLoader(test_dataset, batch_size=num_samples, shuffle=True)
    
    inputs, labels = next(iter(test_loader))
    inputs = inputs.to(device)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(inputs).cpu().numpy()
    
    labels = labels.cpu().numpy()
    
    # Initialize report generator
    generator = ClinicalReportGenerator()
    
    # Generate reports
    reports_dir = os.path.join(config.BASE_DIR, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    all_reports = []
    
    for i in range(num_samples):
        report = generator.generate_report(
            predictions=predictions[i],
            true_labels=labels[i],
            shap_values=None,  # Can add SHAP values if available
            sample_id=i+1
        )
        
        all_reports.append(report)
        
        # Save individual report
        report_path = os.path.join(reports_dir, f'clinical_report_sample_{i+1}.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✓ Generated report for Sample {i+1}")
    
    # Save combined report
    combined_path = os.path.join(reports_dir, 'all_clinical_reports.txt')
    with open(combined_path, 'w', encoding='utf-8') as f:
        f.write("\n\n\n".join(all_reports))
    
    print(f"\n✅ Clinical reports saved to: {reports_dir}")
    print(f"   - Individual reports: clinical_report_sample_1.txt, etc.")
    print(f"   - Combined report: all_clinical_reports.txt")
    
    # Print first report as example
    print("\n" + "="*60)
    print("EXAMPLE CLINICAL REPORT (Sample 1):")
    print("="*60)
    print(all_reports[0])


if __name__ == '__main__':
    generate_GenAI_sample_reports(num_samples=3)
