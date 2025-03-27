
"""
Execution:

$ python -m domain.overnight.lf_interface.example
"""

from overnight.domain_overnight import OvernightDomain, set_evaluator_path


OVERNIGHT_DOMAINS = ['basketball', 'blocks', 'calendar', 'housing', 'publications', 'recipes', 'restaurants', 'socialnetwork']


def test():
    set_evaluator_path('./overnight/evaluator')
    domain = 'basketball'
    assert domain in OVERNIGHT_DOMAINS
    evaluator = OvernightDomain(domain)
    lf_list = ['( call SW.getProperty en.player.kobe_bryant ( call SW.reverse ( string player ) ) )',
               '( call SW.singleton en.team )',
               '( call SW.getProperty ( call SW.singleton en.team ) ( string ! type ) )',  # all objects
               '( call SW.getProperty en.team ( string ! type ) )',  # all objects
               '( call SW.reverse ( string player ) )',
               '( string player )',
               '( call SW.reverse ( string type ) )',
               '( string type )',
               '( call SW.reverse ( call SW.reverse ( string type ) ) )',
               '( call SW.getProperty en.player ( string ! type ) )',  # all objects
               '( call SW.domain ( string player ) )',  # all subjects
               # '( call SW.domain ( string stats ) )',  # It's not working since "stats" is not a property
               '( call SW.getProperty en.stats ( string ! type ) )',
               '( call SW.getProperty en.stats.0 ( string num_points ) )',
               '( call SW.getProperty ( number 4 point ) ( call SW.reverse ( string num_points ) ) )',
               ]    
    denotations = evaluator.execute_logical_forms(lf_list)
    for denotation in denotations:
        print(repr(denotation))


if __name__ == '__main__':
    test()
