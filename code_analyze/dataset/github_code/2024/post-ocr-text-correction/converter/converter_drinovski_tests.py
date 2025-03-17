import os
import re
import sys
import inspect
import unittest
from converter_drinovski import Converter

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from config import ConfigManager

class TestConverter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        config = ConfigManager()
        cls.converter = Converter(config)

    # Тестове на правило 1
    def test_verb_nouns_1(self):
        actual = self.converter.convert_verb_nouns('въздигане', 'въздигане')
        self.assertEqual(actual, ('въздиганье', True), "Expected 'въздиганье'")

    def test_verb_nouns_2(self):
        actual = self.converter.convert_verb_nouns('скачане', 'скачане')
        self.assertEqual(actual, ('скачанье', True), "Expected 'скачанье'")

    def test_verb_nouns_3(self):
        actual = self.converter.convert_verb_nouns('изкачване', 'изкачване')
        self.assertEqual(actual, ('изкачванье', True), "Expected 'изкачванье'")

    # Тестове на правило 2
    def test_ending_1(self):
        actual = self.converter.convert_ending('учител', 'учител')
        self.assertEqual(actual, ('учитель', True), "Expected 'учитель'")

    def test_ending_2(self):
        actual = self.converter.convert_ending('лудост', 'лудост')
        self.assertEqual(actual, ('лудость', True), "Expected 'лудость'")

    def test_ending_3(self):
        actual = self.converter.convert_ending('скромност', 'скромност')
        self.assertEqual(actual, ('скромность', True), "Expected 'скромность'")

    def test_ending_4(self):
        actual = self.converter.convert_ending('светлосин', 'светлосин')
        self.assertEqual(actual, ('светлосинь', True), "Expected 'светлосинь'")

    def test_ending_5(self):
        actual = self.converter.convert_ending('рибар', 'рибар')
        self.assertEqual(actual, ('рибарь', True), "Expected 'рибарь'")

    def test_ending_6(self):
        actual = self.converter.convert_ending('ключар', 'ключар')
        self.assertEqual(actual, ('ключарь', True), "Expected 'ключарь'")

    def test_ending_7(self):
        actual = self.converter.convert_ending('деведесет', 'деведесет') # числена форма
        self.assertEqual(actual, ('деведесеть', True), "Expected 'деведесеть'")

    def test_ending_8(self):
        actual = self.converter.convert_ending('пясък', 'пясък')
        self.assertEqual(actual, ('пясъкъ', True), "Expected 'пясъкъ'")

    def test_ending_9(self):
        actual = self.converter.convert_ending('лък', 'лък')
        self.assertEqual(actual, ('лъкъ', True), "Expected 'лъкъ'")

    def test_ending_10(self):
        actual = self.converter.convert_ending('къща', 'къща')
        self.assertEqual(actual, ('къща', False), "Expected 'къща'")

    # Тест на правило 3
    def test_apply_us_1(self):
        actual = self.converter.apply_us('искаха', 'искаха')
        self.assertEqual(actual, ('искахѫ', True), "Expected 'искахѫ'")

    def test_apply_us_2(self):
        actual = self.converter.apply_us('ходиха', 'ходиха')
        self.assertEqual(actual, ('ходихѫ', True), "Expected 'ходихѫ'")

    def test_apply_us_3(self):
        actual = self.converter.apply_us('винаги', 'винаги')
        self.assertEqual(actual, ('винѫги', True), "Expected 'винѫги'")

    def test_apply_us_4(self):
        actual = self.converter.apply_us('веднъж', 'веднъж')
        self.assertEqual(actual, ('веднѫж', True), "Expected 'веднѫж'")

        actual_2 = self.converter.convert_ending('веднѫж', 'веднѫж')
        self.assertEqual(actual_2, ('веднѫжъ', True), "Expected 'веднѫжъ'")

    # Тестове на правило 4
    def test_convert_the_1(self):
        actual = self.converter.convert_the('мъжът', 'мъжът')
        self.assertEqual(actual, ('мъжътъ', True), "Expected 'мъжътъ'")

    def test_convert_the_2(self):
        actual = self.converter.convert_the('слонът', 'слонът')
        self.assertEqual(actual, ('слонътъ', True), "Expected 'слонътъ'")

    def test_convert_the_3(self):
        actual = self.converter.convert_the('високият', 'високият')
        self.assertEqual(actual, ('високий', True), "Expected 'високий'")

    def test_convert_the_4(self):
        actual = self.converter.convert_the('жената', 'жената')
        self.assertEqual(actual, ('жената', False), "Expected 'жената'") # Why true?

    def test_convert_the_5(self):
        actual = self.converter.convert_the('костта', 'костта')
        self.assertEqual(actual, ('костьта', True), "Expected 'костьта'")

    def test_convert_the_6(self):
        actual = self.converter.convert_the('черешата', 'черешата')
        self.assertEqual(actual, ('черешата', False), "Expected 'черешата'")

    def test_convert_the_7(self):
        actual = self.converter.convert_the('мъжете', 'мъжете')
        self.assertEqual(actual, ('мъжетѣ', True), "Expected 'мъжетѣ'")

    def test_convert_the_8(self):
        actual = self.converter.convert_the('жените', 'жените')
        self.assertEqual(actual, ('женитѣ', True), "Expected 'женитѣ'")

    def test_convert_the_9(self):
        actual = self.converter.convert_the('плочките', 'плочките')
        self.assertEqual(actual, ('плочкитѣ', True), "Expected 'плочкитѣ'")

    def test_convert_the_10(self):
        actual = self.converter.convert_the('кучетата', 'кучетата')
        self.assertEqual(actual, ('кучетатѣ', True), "Expected 'кучетатѣ'")

    def test_convert_the_11(self):
        actual = self.converter.convert_the('шишетата', 'шишетата')
        self.assertEqual(actual, ('шишетатѣ', True), "Expected 'шишетатѣ'")

    def test_convert_the_12(self):
        actual = self.converter.convert_the('хвърчилото', 'хвърчилото')
        self.assertEqual(actual, ('хвърчилото', False), "Expected 'хвърчилото'")

    # Тестове на правило 5 (преобразуване на Ят)
    def test_apply_yat_1(self):
        actual = self.converter.apply_yat('човек', 'човек')
        self.assertEqual(actual, ('човѣк', True), "Expected 'човѣк'")

    def test_apply_yat_2(self):
        actual = self.converter.apply_yat('человек', 'человек')
        self.assertEqual(actual, ('человѣк', True), "Expected 'человѣк'")

    def test_apply_yat_3(self):
        actual = self.converter.apply_yat('цел', 'цел')
        self.assertEqual(actual, ('цѣл', True), "Expected 'цѣл'")

    def test_apply_yat_4(self):
        actual = self.converter.apply_yat('пречките', 'пречките')
        self.assertEqual(actual, ('прѣчките', True), "Expected 'прѣчките'")

    def test_apply_yat_5(self):
        actual = self.converter.apply_yat('пея', 'пея')
        self.assertEqual(actual, ('пѣѭ', True), "Expected 'пѣѭ'")

    def test_apply_yat_6(self):
        actual = self.converter.apply_yat('сее', 'сее')
        self.assertEqual(actual, ('сѣе', True), "Expected 'сѣе'")

    # глаголи мин.св.време
    def test_apply_yat_7(self):
        actual = self.converter.apply_yat('бездействахте', 'бездействахте')
        self.assertEqual(actual, ('бездѣйствахте', True), "Expected 'бездѣйствахте'")
    
    def test_apply_yat_8(self):
        actual = self.converter.apply_yat('намери', 'намери')
        self.assertEqual(actual, ('намѣри', True), "Expected 'намѣри'")
    
    # глаголи мин.несв.време
    def test_apply_yat_9(self):
        actual = self.converter.apply_yat('закъснявахте', 'закъснявахте')
        self.assertEqual(actual, ('закъснѣвахте', True), "Expected 'закъснѣвахте'")
    
    def test_apply_yat_10(self):
        actual = self.converter.apply_yat('известявах', 'известявах')
        self.assertEqual(actual, ('извѣстявах', True), "Expected 'извѣстявах'")

    # Повелително наклонение
    def test_apply_yat_11(self):
        actual = self.converter.apply_yat('съдете', 'съдете')
        self.assertEqual(actual, ('съдѣте', True), "Expected 'съдѣте'")
    
    def test_apply_yat_12(self):
        actual = self.converter.apply_yat('напомнете', 'напомнете')
        self.assertEqual(actual, ('напомнѣте', True), "Expected 'напомнѣте'")

    # наречия завършващи на е
    def test_apply_yat_13(self):
        actual = self.converter.apply_yat('добре', 'добре')
        self.assertEqual(actual, ('добрѣ', True), "Expected 'добрѣ'")
    
    def test_apply_yat_14(self):
        actual = self.converter.apply_yat('после', 'после')
        self.assertEqual(actual, ('послѣ', True), "Expected 'послѣ'")

    def test_apply_yat_15(self):
        actual = self.converter.apply_yat('въобще', 'въобще') # изключение от наречие завършващо на е
        self.assertEqual(actual, ('въобще', False), "Expected 'въобще'")

    # Местоимения
    def test_apply_yat_16(self):
        actual = self.converter.apply_yat('тези', 'тези')
        self.assertEqual(actual, ('тѣзи', True), "Expected 'тѣзи'")
    
    def test_apply_yat_17(self):
        actual = self.converter.apply_yat('онез', 'онез')
        self.assertEqual(actual, ('онѣз', True), "Expected 'онѣз'")

    #представката пре
    def test_apply_yat_18(self):
        actual = self.converter.apply_yat('пребъде', 'пребъде')
        self.assertEqual(actual, ('прѣбъдѣ', True), "Expected 'прѣбъдѣ'")

    def test_apply_yat_19(self):
        actual = self.converter.apply_yat('превзет', 'превзет')
        self.assertEqual(actual, ('прѣвзет', True), "Expected 'прѣвзет'")

    # Тестове на правило 6
    def test_apply_us_5(self):
        actual = self.converter.apply_us('къща', 'къща')
        self.assertEqual(actual, ('кѫща', True), "Expected 'кѫща'")

    def test_apply_us_6(self):
        actual = self.converter.apply_us('лъка', 'лъка')
        self.assertEqual(actual, ('лѫка', True), "Expected 'лѫка'")

    def test_apply_us_7(self):
        actual = self.converter.apply_us('сеят', 'сеят')
        self.assertEqual(actual, ('сеѭт', True), "Expected 'сеѭт'")

    def test_apply_us_8(self):
        actual = self.converter.apply_us('ходя', 'ходя')
        self.assertEqual(actual, ('ходѭ', True), "Expected 'ходѭ'")

    def test_apply_us_9(self):
        actual = self.converter.apply_us('мета', 'мета')
        self.assertEqual(actual, ('метѫ', True), "Expected 'метѫ'")

    def test_apply_us_10(self):
        actual = self.converter.apply_us('видяха', 'видяха')
        self.assertEqual(actual, ('видяхѫ', True), "Expected 'видяхѫ'")

    def test_apply_us_11(self):
        actual = self.converter.apply_us('рекоха', 'рекоха')
        self.assertEqual(actual, ('рекохѫ', True), "Expected 'рекохѫ'")

    def test_apply_us_12(self):
        actual = self.converter.apply_us('нея', 'нея')
        self.assertEqual(actual, ('неѭ', True), "Expected 'неѭ'")

    def test_apply_us_13(self):
        actual = self.converter.apply_us('я', 'я')
        self.assertEqual(actual, ('ѭ', True), "Expected 'ѭ'")

    # Тестове на множество правила върху една дума
    def test_all_1(self):
        actual = self.converter.convert_word('ходя')
        self.assertEqual(actual, 'ходѭ', "Expected 'ходѭ'")

    def test_all_2(self):
        actual = self.converter.convert_word('пресъхнал')
        self.assertEqual(actual, 'прѣсъхналъ', "Expected 'прѣсъхналъ'")

    def test_all_3(self):
        actual = self.converter.convert_word('пресякох')
        self.assertEqual(actual, 'прѣсѣкохъ', "Expected 'прѣсѣкохъ'")

    def test_all_4(self):
        actual = self.converter.convert_word('подсъдимия')
        self.assertEqual(actual, 'подсѫдимия', "Expected 'подсѫдимия'")

    def test_all_5(self):
        actual = self.converter.convert_word('желае')
        self.assertEqual(actual, 'желае', "Expected 'желае'")

    def test_all_6(self):
        actual = self.converter.convert_word('желая')
        self.assertEqual(actual, 'желаѭ', "Expected 'желаѭ'")

    def test_all_7(self):
        actual = self.converter.convert_word('живее')
        self.assertEqual(actual, 'живѣе', "Expected 'живѣе'")

    # Тестове на примерите от преддипломния проект
    # бягане, кост, искаха, пея, царят, новият, летят
    # бѣгане, кость, искаха, пѣя, царьтъ, новиятъ, летятъ
    def test_example_1(self):
        actual = self.converter.convert_word('бягане')
        self.assertEqual(actual, 'бѣганье', "Expected 'бѣганье'")

    def test_example_2(self):
        actual = self.converter.convert_word('кост')
        self.assertEqual(actual, 'кость', "Expected 'кость'")

    def test_example_3(self):
        actual = self.converter.convert_word('искаха')
        self.assertEqual(actual, 'искахѫ', "Expected 'искахѫ'")

    def test_example_4(self):
        actual = self.converter.convert_word('пея')
        self.assertEqual(actual, 'пѣѭ', "Expected 'пѣѭ'")

    def test_example_5(self):
        actual = self.converter.convert_word('царят')
        self.assertEqual(actual, 'царьтъ', "Expected 'царьтъ'")

    def test_example_6(self):
        actual = self.converter.convert_word('новият')
        self.assertEqual(actual, 'новий', "Expected 'новий'")

    def test_example_7(self):
        actual = self.converter.convert_word('летят')
        self.assertEqual(actual, 'летѭтъ', "Expected 'летѭтъ'")

def main():
    unittest.main()

if __name__ == '__main__':
    main()