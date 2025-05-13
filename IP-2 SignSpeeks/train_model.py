import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_save_model():
    try:
        # Load dataset
        print("🔍 Loading dataset...")
        data_dict = pickle.load(open('data.pickle', 'rb'))
        data = data_dict['data']
        labels = data_dict['labels']
        labels_dict = data_dict.get('labels_dict', {})
        
        print(f"📊 Original number of samples: {len(data)}")
        print(f"🎯 Original number of classes: {len(set(labels))}")
        
        # Print original class distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        print("\n📈 Original Class Distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"{label}: {count} samples")

        # Remove classes with too few samples
        MIN_SAMPLES = 5
        valid_classes = [label for label, count in zip(unique_labels, counts) if count >= MIN_SAMPLES]
        print(f"\n🔄 Removing classes with fewer than {MIN_SAMPLES} samples...")
        print(f"❌ Removed classes: {[label for label in unique_labels if label not in valid_classes]}")
        
        # Filter data and labels
        mask = np.isin(labels, valid_classes)
        data = np.array(data)[mask]
        labels = np.array(labels)[mask]
        
        print(f"\n📊 Filtered number of samples: {len(data)}")
        print(f"🎯 Filtered number of classes: {len(set(labels))}")
        
        # Print new class distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        print("\n📈 New Class Distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"{label}: {count} samples")

        print(f"\n📏 Feature dimension: {data.shape[1]}")

        # Scale the features
        print("🔄 Scaling features...")
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        # Train-test split with stratification
        print("\n📊 Splitting data into train and test sets...")
        x_train, x_test, y_train, y_test = train_test_split(
            data_scaled, labels, test_size=0.2, stratify=labels, random_state=42
        )

        # Initialize and train Random Forest model
        print("\n🤖 Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight='balanced',  # Handle class imbalance
            random_state=42
        )
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, x_train, y_train, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train on full training set
        model.fit(x_train, y_train)
        
        # Evaluate on test set
        y_predict = model.predict(x_test)
        score = accuracy_score(y_predict, y_test)
        print(f"Test Set Accuracy: {score * 100:.2f}%")

        # Print detailed classification report
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_predict))

        # Save model and metadata
        print("\n💾 Saving model and metadata...")
        model_data = {
            'model': model,
            'labels_dict': labels_dict,
            'scaler': scaler,
            'valid_classes': valid_classes  # Save the list of valid classes
        }

        with open('model.p', 'wb') as f:
            pickle.dump(model_data, f)

        print("✅ Model saved successfully in model.p")
        
        # Plot feature importance
        print("\n📊 Generating feature importance plot...")
        feature_importance = model.feature_importances_
        plt.figure(figsize=(12, 6))
        sns.barplot(x=range(len(feature_importance)), y=feature_importance)
        plt.title('Feature Importance')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        print("✅ Feature importance plot saved as feature_importance.png")

    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    train_and_save_model() 