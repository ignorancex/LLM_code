import os
import re
import sys
import inspect
import unittest
from converter_ivanchevski import Converter

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from config import ConfigManager

class TestConverter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        config = ConfigManager()
        cls.converter = Converter(config)

    # Тестове за проверка на краесловните ерове
    def test_ending_conversion_1(self):
        actual = self.converter.check_ending('цар', 'цар')
        self.assertEqual(actual, ('царь', True), "Expected 'царь'")

    def test_ending_conversion_2(self):
        actual = self.converter.check_ending('нокът', 'нокът')
        self.assertEqual(actual, ('нокъть', True), "Expected 'нокъть'")

    def test_ending_conversion_3(self):
        actual = self.converter.check_ending('спасител', 'спасител')
        self.assertEqual(actual, ('спаситель', True), "Expected 'спаситель'")

    def test_ending_conversion_4(self):
        actual = self.converter.check_ending('камък', 'камък')
        self.assertEqual(actual, ('камъкъ', True), "Expected 'камъкъ'")

    def test_ending_conversion_5(self):
        actual = self.converter.check_ending('риба', 'риба')
        self.assertEqual(actual, ('риба', False), "Expected 'риба'")

    # Тестове за проверка на голям юс (ѫ)
    def test_us_1(self):
        actual = self.converter.check_us('къща', 'къща')
        self.assertEqual(actual, ('кѫща', True), "Expected 'кѫща'")

    def test_us_2(self):
        actual = self.converter.check_us('мъж', 'мъж')
        self.assertEqual(actual, ('мѫж', True), "Expected 'мѫж'")

    def test_us_3(self):
        actual = self.converter.check_us('са', 'са')
        self.assertEqual(actual, ('сѫ', True), "Expected 'сѫ'")

    def test_us_4(self):
        actual = self.converter.check_us('къде', 'къде')
        self.assertEqual(actual, ('кѫде', True), "Expected 'кѫде'")

    def test_us_5(self):
        actual = self.converter.check_us('сън', 'сън')
        self.assertEqual(actual, ('сън', False), "Expected 'сън'")

    # Тестове за проверка на ят (ѣ)
    def test_yat_1(self):
        actual = self.converter.check_yat('мляко', 'мляко')
        self.assertEqual(actual, ('млѣко', True), "Expected 'млѣко'")

    def test_yat_2(self):
        actual = self.converter.check_yat('дете', 'дете')
        self.assertEqual(actual, ('детѣ', True), "Expected 'детѣ'")

    def test_yat_3(self):
        actual = self.converter.check_yat('подире', 'подире')
        self.assertEqual(actual, ('подирѣ', True), "Expected 'подирѣ'")

    def test_yat_4(self):
        actual = self.converter.check_yat('тези', 'тези')
        self.assertEqual(actual, ('тѣзи', True), "Expected 'тѣзи'")

    def test_yat_5(self):
        actual = self.converter.check_yat('после', 'после')
        self.assertEqual(actual, ('послѣ', True), "Expected 'послѣ'")

    def test_yat_6(self):
        actual = self.converter.check_yat('въобще', 'въобще')
        self.assertEqual(actual, ('въобще', False), "Expected 'въобще'")

    # Тестове за проверка на във/със
    def test_vs(self):
        actual = self.converter.check_vs('във', 'във')
        self.assertEqual(actual, ('въ', True), "Expected 'въ'")

    def test_vs(self):
        actual = self.converter.check_vs('със', 'със')
        self.assertEqual(actual, ('съ', True), "Expected 'съ'")

     # Добавя буква "ь" при членуването на съществителни от женски род където има двойно "т"
    def test_check_feminine_the_1(self):
        actual = self.converter.check_feminine_the('хубостта', 'хубостта')
        self.assertEqual(actual, ('хубостьта', True), "Expected 'хубостьта'")

    def test_check_feminine_the_2(self):
        actual = self.converter.check_feminine_the('пролетта', 'пролетта')
        self.assertEqual(actual, ('пролетьта', True), "Expected 'пролетьта'")

    # Специални думи
    def test_exclusion_words_1(self):
        actual = self.converter.check_exclusion_words_1('сърце', 'сърце')
        self.assertEqual(actual, ('сърдце', True), "Expected 'сърдце'")

    def test_exclusion_words_2(self):
        actual = self.converter.check_exclusion_words_1('празник', 'празник')
        self.assertEqual(actual, ('праздник', True), "Expected 'праздник'")

    def test_exclusion_words_3(self):
        actual = self.converter.check_exclusion_words_2('овошкa', 'овошкa')
        self.assertEqual(actual, ('овощкa', True), "Expected 'овощкa'")

    # Тестове на примерите от преддипломния проект
    # бягане, кост, искаха, пея, царят, новият, летят
    # бѣгане, кость, искаха, пѣя, царьтъ, новиятъ, летятъ
    def test_example_1(self):
        actual = self.converter.convert_words(['бягане'])
        self.assertEqual(actual, 'бѣгане', "Expected 'бѣгане'")

    def test_example_2(self):
        actual = self.converter.convert_words(['кост'])
        self.assertEqual(actual, 'кость', "Expected 'кость'")

    def test_example_3(self):
        actual = self.converter.convert_words(['искаха'])
        self.assertEqual(actual, 'искаха', "Expected 'искаха'")

    def test_example_4(self):
        actual = self.converter.convert_words(['пея'])
        self.assertEqual(actual, 'пѣя', "Expected 'пѣя'")

    def test_example_5(self):
        actual = self.converter.convert_words(['новият'])
        self.assertEqual(actual, 'новиятъ', "Expected 'новиятъ'")

    def test_example_6(self):
        actual = self.converter.convert_words(['летят'])
        self.assertEqual(actual, 'летятъ', "Expected 'летятъ'")

def main():
    unittest.main()
    
if __name__ == '__main__':
    main()