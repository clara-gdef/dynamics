rsync -ravh --exclude '.idea' --exclude '*/__pycache__'  --exclude '.git'  --exclude '*/img'  --exclude 'models/logs' ./ gainondefor@gate.lip6.fr:Documents/code/dynamics/