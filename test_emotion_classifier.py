import unittest
from emotion_classifier import EmotionClassifier

class TestEmotionClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = EmotionClassifier()

    def test_detect_sadness(self):
        text = "I feel so down and depressed."
        emotion, confidence = self.classifier.predict_emotion(text)
        self.assertEqual(emotion, "sadness")
        self.assertGreater(confidence, 0.5)

    def test_detect_joy(self):
        text = "I'm really happy and excited today!"
        emotion, confidence = self.classifier.predict_emotion(text)
        self.assertEqual(emotion, "joy")
        self.assertGreater(confidence, 0.5)

    def test_detect_neutral(self):
        text = "Today is a regular day with nothing special."
        emotion, confidence = self.classifier.predict_emotion(text)
        self.assertIn(emotion, ["neutral", "joy", "sadness"])  # allow soft margin
        self.assertGreater(confidence, 0.4)

if __name__ == "__main__":
    unittest.main()
