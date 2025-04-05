
class FeatureManager:
    def __init__(self):
        self.featureList = {}

    def __enter__(self):
        print "Initializing mfcc calculation..."

    def __exit__(self, exc_type, exc_val, exc_tb):
        print "Done with calculations..."

    def addRegisteredFeatures(self, feature, featureId):
        self.featureList[featureId] = feature

    def getRegisteredFeatures(self):
        return self.featureList

    def getRegisteredFeature(self, feature_id):
        return self.featureList[feature_id]

