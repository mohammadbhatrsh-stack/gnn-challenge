"""
scoring_script.py
Challenge Scoring Script - Participants will submit solutions that work with this script
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
import networkx as nx
import numpy as np
import json
import time
import sys
import os

# Import the participant's solution
try:
    from submission import EnhancedGraphModel, compute_enhanced_features
    print("✓ Successfully imported participant's solution")
except ImportError:
    print("✗ Error: Could not import submission.py")
    print("Please ensure submission.py contains:")
    print("  - EnhancedGraphModel class")
    print("  - compute_enhanced_features function")
    sys.exit(1)

class ChallengeEvaluator:
    """Evaluates participant submissions on hidden test data"""
    
    def __init__(self, test_mode=False):
        self.test_mode = test_mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load evaluation dataset (different from training)
        self.load_evaluation_dataset()
        
        # Scoring weights
        self.weights = {
            'accuracy': 0.7,
            'efficiency': 0.2,
            'novelty': 0.1
        }
    
    def load_evaluation_dataset(self):
        """Load hidden evaluation dataset"""
        # In practice, this would load a separate test set
        # For the challenge, we'll use a mix of datasets
        from torch_geometric.datasets import TUDataset
        
        # Create complex evaluation set
        datasets = []
        
        # Multiple graph types to test generalization
        for name in ['MUTAG', 'PROTEINS', 'NCI1']:
            try:
                dataset = TUDataset(root='/tmp/challenge_data', name=name)
                datasets.append(dataset)
            except:
                continue
        
        # Combine and shuffle
        self.eval_data = []
        for dataset in datasets:
            for data in dataset:
                if data.num_nodes > 0:  # Filter invalid graphs
                    self.eval_data.append(data)
        
        np.random.shuffle(self.eval_data)
        
        # Use only subset for fair comparison
        self.eval_data = self.eval_data[:500]
        print(f"Loaded {len(self.eval_data)} evaluation graphs")
    
    def compute_features(self, data):
        """Apply participant's feature computation"""
        try:
            features = compute_enhanced_features(data)
            return features
        except Exception as e:
            print(f"Error in compute_enhanced_features: {e}")
            # Fallback to baseline
            G = nx.Graph()
            edge_index_np = data.edge_index.cpu().numpy()
            G.add_edges_from(edge_index_np.T)
            degrees = [G.degree(i) for i in range(data.num_nodes)]
            return torch.tensor(degrees, dtype=torch.float).unsqueeze(1)
    
    def prepare_data(self):
        """Prepare data with participant's features"""
        prepared_data = []
        
        for data in self.eval_data:
            data = data.clone()
            
            # Compute features
            features = self.compute_features(data)
            
            # Combine with original features
            if data.x is None:
                data.x = features
            elif features is not None:
                data.x = torch.cat([data.x, features], dim=1)
            
            prepared_data.append(data)
        
        return prepared_data
    
    def evaluate_accuracy(self, model, data_loader):
        """Evaluate classification accuracy"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                out = model(data)
                pred = out.argmax(dim=1)
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
        
        return correct / total if total > 0 else 0
    
    def evaluate_efficiency(self, model, sample_data):
        """Evaluate computational efficiency"""
        model.eval()
        
        # Time feature computation
        start_time = time.time()
        features = self.compute_features(sample_data)
        feature_time = time.time() - start_time
        
        # Time inference
        with torch.no_grad():
            data = sample_data.clone()
            if data.x is None:
                data.x = torch.ones(data.num_nodes, 1)
            if features is not None:
                data.x = torch.cat([data.x, features], dim=1)
            
            data = data.to(self.device)
            model = model.to(self.device)
            
            start_time = time.time()
            for _ in range(10):  # Multiple runs for stable timing
                _ = model(data)
            inference_time = (time.time() - start_time) / 10
        
        return {
            'feature_time_ms': feature_time * 1000,
            'inference_time_ms': inference_time * 1000,
            'total_time_ms': (feature_time + inference_time) * 1000
        }
    
    def evaluate_novelty(self, model):
        """Evaluate solution novelty"""
        novelty_score = 0
        
        # Check model complexity
        num_params = sum(p.numel() for p in model.parameters())
        if num_params < 10000:
            novelty_score += 0.3  # Efficient model
        
        # Check feature usage
        if hasattr(model, 'feature_analysis'):
            novelty_score += 0.3  # Includes analysis
        
        # Check for innovative approaches
        model_str = str(model)
        innovative_terms = ['attention', 'transformer', 'skip', 'residual', 'ensemble']
        for term in innovative_terms:
            if term in model_str.lower():
                novelty_score += 0.1
        
        return min(novelty_score, 1.0)
    
    def run_evaluation(self):
        """Run complete evaluation"""
        print("\n" + "="*60)
        print("EVALUATING SUBMISSION")
        print("="*60)
        
        # Prepare data
        print("\n1. Preparing data with enhanced features...")
        prepared_data = self.prepare_data()
        data_loader = DataLoader(prepared_data, batch_size=32, shuffle=False)
        
        # Initialize model
        print("2. Initializing enhanced model...")
        try:
            sample = prepared_data[0]
            input_dim = sample.x.shape[1]
            model = EnhancedGraphModel(input_dim).to(self.device)
        except Exception as e:
            print(f"Error initializing model: {e}")
            return None
        
        # Train model (limited epochs for evaluation)
        print("3. Training model...")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Quick training for evaluation
        for epoch in range(10):
            model.train()
            for data in data_loader:
                data = data.to(self.device)
                optimizer.zero_grad()
                out = model(data)
                loss = F.nll_loss(out, data.y)
                loss.backward()
                optimizer.step()
        
        # Run evaluations
        print("4. Running evaluations...")
        
        # Accuracy
        accuracy = self.evaluate_accuracy(model, data_loader)
        print(f"   Accuracy: {accuracy:.4f}")
        
        # Efficiency
        sample_data = prepared_data[0]
        efficiency = self.evaluate_efficiency(model, sample_data)
        print(f"   Feature computation: {efficiency['feature_time_ms']:.2f} ms")
        print(f"   Inference: {efficiency['inference_time_ms']:.2f} ms")
        
        # Efficiency score (lower is better, normalized)
        total_time = efficiency['total_time_ms']
        efficiency_score = max(0, 1 - total_time / 100)  # Target < 100ms
        
        # Novelty
        novelty_score = self.evaluate_novelty(model)
        print(f"   Novelty score: {novelty_score:.3f}")
        
        # Calculate final score
        final_score = (
            self.weights['accuracy'] * accuracy +
            self.weights['efficiency'] * efficiency_score +
            self.weights['novelty'] * novelty_score
        )
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Accuracy:           {accuracy:.4f} (weight: {self.weights['accuracy']})")
        print(f"Efficiency Score:   {efficiency_score:.4f} (weight: {self.weights['efficiency']})")
        print(f"Novelty Score:      {novelty_score:.4f} (weight: {self.weights['novelty']})")
        print(f"\nFINAL SCORE:        {final_score:.4f}")
        print("="*60)
        
        # Save results
        results = {
            'accuracy': float(accuracy),
            'efficiency_score': float(efficiency_score),
            'novelty_score': float(novelty_score),
            'final_score': float(final_score),
            'efficiency_details': efficiency,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open('evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\nResults saved to 'evaluation_results.json'")
        
        return final_score

def main():
    """Main evaluation script"""
    evaluator = ChallengeEvaluator()
    score = evaluator.run_evaluation()
    
    if score is not None:
        print(f"\nSubmit your score: {score:.4f}")
        print("Use the submission form to upload your solution and results!")
    
    return score

if __name__ == "__main__":
    main()
