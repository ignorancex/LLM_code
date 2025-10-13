# Dictionary to store the criticality scores based on check names
criticality_scores = {
    "1.1.1": 5, "1.1.2": 5, "1.1.3": 4, "1.1.4": 4, "1.1.5": 3, "1.1.6": 3, "1.1.7": 5, "1.1.8": 5, "1.1.9": 3,
    "1.1.10": 3, "1.1.11": 5, "1.1.12": 5, "1.1.13": 4, "1.1.14": 4, "1.1.15": 3, "1.1.16": 3, "1.1.17": 4, 
    "1.1.18": 4, "1.1.19": 5, "1.1.20": 4, "1.1.21": 5, "1.2.1": 5, "1.2.2": 5, "1.2.3": 5, "1.2.4": 4, "1.2.5": 4, 
    "1.2.6": 4, "1.2.7": 5, "1.2.8": 4, "1.2.9": 5, "1.2.10": 3, "1.2.11": 5, "1.2.12": 3, "1.2.13": 4, "1.2.14": 4,
    "1.2.15": 3, "1.2.16": 5, "1.2.17": 4, "1.2.18": 5, "1.2.19": 5, "1.2.20": 5, "1.2.21": 3, "1.2.22": 5, "1.2.23": 4,
    "1.2.24": 4, "1.2.25": 4, "1.2.26": 3, "1.2.27": 5, "1.2.28": 4, "1.2.29": 5, "1.2.30": 5, "1.2.31": 5, "1.2.32": 5,
    "1.2.33": 4, "1.2.34": 5, "1.2.35": 5, "1.3.1": 2, "1.3.2": 3, "1.3.3": 4, "1.3.4": 5, "1.3.5": 5, "1.3.6": 5,
    "1.3.7": 4, "1.4.1": 3, "1.4.2": 4, "2.1": 5, "2.2": 5, "2.3": 5, "2.4": 5, "2.5": 5, "2.6": 5, "2.7": 5, 
    "4.1.1": 4, "4.1.2": 4, "4.1.3": 3, "4.1.4": 3, "4.1.5": 4, "4.1.6": 4, "4.1.7": 4, "4.1.8": 4, "4.1.9": 4, 
    "4.1.10": 4, "4.2.1": 5, "4.2.2": 5, "4.2.3": 5, "4.2.4": 4, "4.2.5": 3, "4.2.6": 5, "4.2.7": 4, "4.2.8": 3,
    "4.2.9": 3, "4.2.10": 5, "4.2.11": 5, "4.2.12": 5, "4.2.13": 5, "5.1.1": 5, "5.1.2": 5, "5.1.3": 4, "5.1.4": 5,
    "5.1.5": 4, "5.1.6": 4, "5.2.1": 5, "5.2.2": 4, "5.2.3": 4, "5.2.4": 4, "5.2.5": 4, "5.2.6": 5, "5.2.7": 4,
    "5.2.8": 4, "5.2.9": 4, "5.3.1": 4, "5.3.2": 4, "5.4.1": 5, "5.4.2": 5, "5.5.1": 4, "5.7.1": 3, "5.7.2": 4,
    "5.7.3": 5, "5.7.4": 3, "3.1.1": 3, "3.2.1": 4, "3.2.2": 4
}

# Define the coefficients for each check mark
coefficients = {
    '[PASS]': 1,
    '[FAIL]': -1,
    '[WARN]': 0.5
}

# Function to calculate the score based on the file content
def calculate_score(file_path):
    total_score = 0
    
    with open(file_path, 'r') as file:
        for line in file:
            # Check for [PASS], [FAIL], or [WARN] in the line
            for mark in coefficients.keys():
                if mark in line:
                    # Extract the check ID (e.g., "1.1.1") before the mark
                    check_id = line.split(' ')[1]
                    if check_id in criticality_scores:
                        # Calculate the score (criticality * coefficient) and add it to total
                        criticality = criticality_scores[check_id]
                        score = criticality * coefficients[mark]
                        total_score += score
    
    return total_score

# Usage: Replace 'kube_bench_report.txt' with your actual file path
distValues = ['k0s', 'k3s', 'k8s', 'KubeEdge', 'OpenYurt']

for dist in distValues:
    file_path = f'./{dist}-kube-bench.report'
    result = calculate_score(file_path)
    print(f"{dist} Total Score: {result} points")
    max_total_score = 610
    normalized_score = result * 100 / max_total_score
    print(f"{dist} Normalized Score: {normalized_score:.2f}%")
    

# Output:
# k0s Total Score: 144.5 points
# k0s Normalized Score: 23.69%
# k3s Total Score: 44.0 points
# k3s Normalized Score: 7.21%
# k8s Total Score: 335.5 points
# k8s Normalized Score: 55.00%
# KubeEdge Total Score: 335.5 points
# KubeEdge Normalized Score: 55.00%
# OpenYurt Total Score: 335.5 points
# OpenYurt Normalized Score: 55.00%
