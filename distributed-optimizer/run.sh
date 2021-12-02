# Initialize Comet Optimizer
kubectl apply -f ./job-initialize-comet-optimizer.yaml

# Check that the Comet Optimizer had been created
kubectl wait --for=condition=complete job/initialize-comet-optimizer

# Get optimizer ID
OPTIMIZER_ID=$(kubectl logs --tail=1 job/initialize-comet-optimizer | cut -d "=" -f2)
echo $OPTIMIZER_ID

# Copy job optimizer yaml template
cp job-optimizer.yaml job-optimizer-uuid.yaml

# Replace "REPLACE_WITH_OPTIMIZER_ID" with Optimizer ID in job-optimizer-uuid.yaml
sed -i '' -e "s/REPLACE_WITH_OPTIMIZER_ID/$OPTIMIZER_ID/g" job-optimizer-uuid.yaml

# Run the Optimization process
kubectl apply -f ./job-optimizer-uuid.yaml
