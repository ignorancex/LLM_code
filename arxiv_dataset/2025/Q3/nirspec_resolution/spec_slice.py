def get_spectra_slice_specs(disperser, mode="MSA", extend=False):
    """Get the spectra slice specifications for the given disperser and mode.

    :param disperser: Disperser name
    :type disperser: str
    :param mode: Mode of the instrument, defaults to "MSA"
    :type mode: str, optional
    :param extend: Whether to send additional lines covered by F070LP wavelengths
    :type extend: bool, optional
    :return: List of dictionaries containing line specifications
    :rtype: list
    """
    disperser = disperser.upper()

    # assert disperser in ["G140", "G235", "G395"]

    if "G140" in disperser:
        line_library = [
            # Pa delta
            {
                "min_lambda": 10020,
                "max_lambda": 10100,
                "main_line_name": r"Pa$\delta$",
                "line_names": ["He ɪ", "He ɪ", "H ɪ", "He ɪ"],
                "line_wavlengths": [
                    10030.469746750001,
                    10033.900158100001,
                    10052.128,
                    10074.806682083332,
                ],
            },
            # He ɪ 10830
            {
                "min_lambda": 10780,
                "max_lambda": 10900,
                "main_line_name": r"He ɪ",
                "line_names": ["He ɪ", "He ɪ", "He ɪ"],
                "line_wavlengths": [
                    10832.057471999999,
                    10833.216751,
                    10833.306444,
                ],
            },
            # Pa gamma
            {
                "min_lambda": 10910,
                "max_lambda": 10990,
                "main_line_name": r"Pa$\gamma$",
                "line_names": ["He ɪ", "He ɪ", "H ɪ"],
                "line_wavlengths": [
                    10915.991686366666,
                    10920.0525431,
                    10941.09,
                ],
            },
            # Pa beta
            {
                "min_lambda": 12760,
                "max_lambda": 12885,
                "main_line_name": r"Pa$\beta$",
                "line_names": ["He ɪ", "He ɪ", "H ɪ", "He ɪ", "He ɪ"],
                "line_wavlengths": [
                    12788.43071063333,
                    12793.999402500001,
                    12821.59,
                    12849.46706775,
                    12849.941156,
                ],
            },
            # Br zeta
            {
                "min_lambda": 17350,
                "max_lambda": 17430,
                "main_line_name": r"Br$\zeta$",
                "line_names": ["He ɪ", "He ɪ", "H ɪ", "He ɪ"],
                "line_wavlengths": [
                    17356.476694124285,
                    17358.26905142667,
                    17366.85,
                    17379.688902016667,
                ],
            },
            # # Br epsilon
            # {
            #     "min_lambda": 18150,
            #     "max_lambda": 18215,
            #     "main_line_name": r"Br $\epsilon$",
            #     "line_names": ["He ɪ", "He ɪ", "He ɪ", "H ɪ"],  # "He ɪ"],
            #     "line_wavlengths": [
            #         18143.99662182,
            #         18168.09013685,
            #         18170.770895771668,
            #         18179.084,
            #         # 18212.142581133332,
            #     ],
            # },
            # Pa alpha
            {
                "min_lambda": 18675,
                "max_lambda": 18820,
                "main_line_name": r"Pa$\alpha$",
                "line_names": ["He ɪ", "He ɪ", "H ɪ"],  # "Fe I",
                "line_wavlengths": [
                    18690.442142149997,
                    18702.3180723,
                    # 18726.374,
                    18756.13,
                ],
            },
        ]

        if mode == "FS":
            line_library.pop(-1)  # Remove Pa alpha for MSA mode
        elif mode == "MSA":
            line_library.pop(-1)
            line_library.pop(1)

        if extend:
            extended_lines = [
                # Pa eta
                {
                    "min_lambda": 8980,
                    "max_lambda": 9060,
                    "main_line_name": r"Pa$\eta$",
                    "line_names": [
                        "He ɪ",
                        # "He ɪ",
                        "He ɪ",
                        # "He ɪ",
                        "He ɪ",
                        "He II",
                        "H ɪ",
                        # "He ɪ",
                    ],
                    "line_wavlengths": [
                        8999.4379225875,
                        # 8999.437558571428,
                        # 8999.4744687,
                        8999.9904730,
                        # 9002.2077345,
                        9011.63606005,
                        9013.688,
                        9017.385,
                        # 9065.816658166667,
                    ],
                },
                # Pa zeta
                {
                    "min_lambda": 9190,
                    "max_lambda": 9290,
                    "main_line_name": r"Pa$\zeta$",
                    "line_names": [
                        "He ɪ",
                        # "He I",
                        "He I",
                        "He I",
                        "H I",
                    ],
                    "line_wavlengths": [
                        9212.852746244444,
                        9215.75788475,
                        9227.763,
                        # 9230.4031705,
                        9231.547,
                    ],
                },
                # Pa epsilon
                # {
                #     "min_lambda": 9510,
                #     "max_lambda": 9590,
                #     "main_line_name": r"Pa$\epsilon$",
                #     "line_names": ["He ɪ", "He ɪ", "He ɪ", "He I", "H ɪ", "He I"],
                #     "line_wavlengths": [
                #         9519.3168135,
                #         9527.0454912,
                #         9528.7639748125,
                #         9531.8764534,
                #         9544.676,
                #         9548.590,
                #         9555.530404233334,
                #     ],
                # },
            ]
            line_library = extended_lines + line_library

            line_library.pop(-2)
            line_library.pop(-1)

    elif "G235" in disperser:
        line_library = [
            # Br eta
            {
                "min_lambda": 16800,
                "max_lambda": 16850,
                "main_line_name": r"Br$\eta$",
                "line_names": ["He ɪ", "He ɪ", "H ɪ"],  # "He ɪ"],
                "line_wavlengths": [
                    16801.156549285715,
                    16802.423072166664,
                    16811.111,
                    # 16812.280147333335,
                ],
            },
            # Br zeta
            {
                "min_lambda": 17300,
                "max_lambda": 17440,
                "main_line_name": r"Br$\zeta$",
                # "lines": [
                #     ["He ɪ", 10832.057471999999],
                #     ["He ɪ", 10833.216751],
                #     ["He ɪ", 10833.306444],
                # ],
                "line_names": ["He ɪ", "He ɪ", "He ɪ", "H ɪ", "He ɪ"],
                "line_wavlengths": [
                    17340.3448365,
                    17356.476694124285,
                    17358.26905142667,
                    17366.85,
                    17379.688902016667,
                ],
            },
            # Pa alpha
            {
                "min_lambda": 18655,
                "max_lambda": 18840,
                "main_line_name": r"Pa$\alpha$",
                "line_names": ["He ɪ", "He ɪ", "H ɪ"],
                "line_wavlengths": [
                    18690.442142149997,
                    18702.3180723,
                    18756.13,
                ],
            },
            # He ɪ 20580
            {
                "min_lambda": 20555,
                "max_lambda": 20660,
                "main_line_name": r"He ɪ",
                "line_names": ["He ɪ", "He ɪ", "He ɪ"],
                "line_wavlengths": [
                    20586.904629999997,
                    20592.79265,
                    20607.463187666668,
                    # 20622.81917,
                ],
            },
            # Br gamma
            {
                "min_lambda": 21575,
                "max_lambda": 21750,
                "main_line_name": r"Br$\gamma$",
                "line_names": ["He ɪ", "He ɪ", "He ɪ", "He ɪ", "He ɪ", "H ɪ"],
                "line_wavlengths": [
                    # 21586.002015,
                    21613.70985955,
                    21622.905790999997,
                    # 21633.575576666666,
                    21647.428962849997,
                    # 21653.3130708,
                    # 21655.37832995,
                    21661.199999999997,
                ],
            },
            # Br beta
            {
                "min_lambda": 26180,
                "max_lambda": 26350,
                "main_line_name": r"Br$\beta$",
                "line_names": ["He ɪ", "He ɪ", "He ɪ", "H ɪ"],
                "line_wavlengths": [
                    26192.12167798333,
                    26205.6152478,
                    26240.915984385712,
                    # 26254.484925,
                    26258.670000000002,
                    # 26258.90785,
                    # 26259.197858133328,
                ],
            },
            # Pfund eta
            {
                "min_lambda": 30350,
                "max_lambda": 30480,
                "main_line_name": r"Pf$\eta$",
                "line_names": [
                    "He ɪ",
                    "He ɪ",
                    "He ɪ",
                    "He ɪ",
                    # "He ɪ",
                    # "He ɪ",
                    "He ɪ",
                    "H ɪ",
                    # "He ɪ",
                ],
                "line_wavlengths": [
                    30373.882203465,
                    30373.934701927996,
                    # 30377.975075399998,
                    # 30378.340039328574,
                    # 30379.09099256333,
                    30379.13724103666,
                    # 30379.406177666668,
                    30392.022,
                    # 30399.158583,
                ],
            },
        ]

        if mode == "FS":
            line_library.pop(0)
            line_library.pop(-1)
        elif mode == "MSA":
            line_library.pop(0)
            line_library.pop(2)

    elif "G395" in disperser:
        line_library = [
            # Pfund eta
            {
                "min_lambda": 30330,
                "max_lambda": 30520,
                "main_line_name": r"Pf$\eta$",
                "line_names": [
                    "He ɪ",
                    "He ɪ",
                    "He ɪ",
                    "He ɪ",
                    # "He ɪ",
                    "He ɪ",
                    "He ɪ",
                    "H ɪ",
                    "He ɪ",
                ],
                "line_wavlengths": [
                    30337.957093580004,
                    30338.043565500004,
                    30338.14237,
                    # 30373.882203465,
                    30373.934701927996,
                    30377.975075399998,
                    # 30378.340039328574,
                    # 30379.09099256333,
                    # 30379.13724103666,
                    30379.406177666668,
                    30392.022,
                    30399.158583,
                    # 30463.121334,
                    # 30476.9185955,
                    #########
                    # 30318.737946666664,
                    # 30323.248917,
                    # 30337.957093580004,
                    # 30338.043565500004,
                    # 30338.14237,
                    # 30348.425645000003,
                    # 30357.262333333332,
                    # 30373.882203465,
                    # 30374.60809750666,
                    # 30378.340039328574,
                    # 30379.09590704,
                    # 30379.120119029998,
                    # 30379.13828167,
                    # 30379.14502218,
                    # 30379.378297433334,
                    # 30379.4340579,
                    # 30392.022,
                    # # 30399.15858,
                    # # 30463.121334,
                    # # 30476.9185955,
                ],
            },
            # Pfund gamma
            {
                "min_lambda": 37350,
                "max_lambda": 37580,
                "main_line_name": r"Pf$\gamma$",
                "line_names": [
                    "He ɪ",
                    "He ɪ",
                    "He ɪ",
                    "H ɪ",
                    # "He ɪ",
                    # "He ɪ",
                    "He ɪ",
                    "He ɪ",
                    "He ɪ",
                    "He ɪ",
                ],
                "line_wavlengths": [
                    37328.38483621667,
                    37381.94650391429,
                    37390.70984881429,
                    # 37397.99492416667,
                    37405.56,
                    37411.76899333333,
                    37412.07705,
                    # 37483.20428666667,
                    37478.342826,
                    37493.933000000005,
                    ####
                    # 37328.36403902,
                    # 37328.4888222,
                    # 37344.1973275,
                    # 37381.94650391429,
                    # 37388.436235214285,
                    # 37390.66808346,
                    # 37390.72655495601,
                    # 37393.75726,
                    # 37397.99492416667,
                    # 37405.56,
                    # 37411.614965,
                    # 37412.07705,
                    # 37478.342826,
                    # 37493.933000000005,
                ],
            },
            # Humphrey iota
            # {
            #     "min_lambda": 39000,
            #     "max_lambda": 39230,
            #     "main_line_name": r"Hu $\iota$",
            #     "line_names": [
            #         "He ɪ",
            #         "H ɪ",
            #         "He ɪ",
            #         "He ɪ",
            #     ],
            #     "line_wavlengths": [
            #         39056.36962333334,
            #         39075.486000000004,
            #         39085.84986333333,
            #         39094.736858,
            #     ],
            # },
            # Humphrey kappa
            # {
            #     "min_lambda": 38160,
            #     "max_lambda": 38300,
            #     "main_line_name": r"Hu $\kappa$",
            #     "line_names": [
            #         "He ɪ",
            #         "He ɪ",
            #         "H ɪ",
            #     ],
            #     "line_wavlengths": [38173.262782, 38178.64426714287, 38194.512],
            # },
            # Brackett alpha
            {
                "min_lambda": 40300,
                "max_lambda": 40730,
                "main_line_name": r"Br$\alpha$",
                "line_names": [
                    "He ɪ",
                    "He ɪ",
                    "He ɪ",
                    "He ɪ",
                    "He ɪ",
                    "H ɪ",
                    "He ɪ",
                    "He ɪ",
                    "He ɪ",
                    "He ɪ",
                ],
                "line_wavlengths": [
                    40377.31950816667,
                    40391.224109999996,
                    40409.4473965,
                    40490.157808985714,
                    40512.137976000005,
                    40522.62,
                    40544.661538,
                    40545.048049000005,
                    40563.50551566667,
                    40574.5684,
                ],
            },
            # He ɪ
            {
                "min_lambda": 42900,
                "max_lambda": 43100,
                "main_line_name": r"He ɪ",
                "line_names": ["He ɪ", "He ɪ", "He ɪ", "He ɪ", "He ɪ", "He ɪ"],
                #         42954.167],
                # ['He ɪ', 42959.1611],
                # ['He ɪ', 42959.566699999996],
                # ['He ɪ', 42959.90127],
                # ['He ɪ', 42959.905360000004],
                # ['He ɪ', 42959.90569
                "line_wavlengths": [
                    42954.167,
                    42959.1611,
                    42959.566699999996,
                    42959.90127,
                    42959.905360000004,
                    42959.90569,
                ],
            },
            # Humphrey Zeta
            {
                "min_lambda": 43725,
                "max_lambda": 43900,
                "main_line_name": r"Hu$\zeta$",
                "line_names": [
                    "He ɪ",
                    "He ɪ",
                    "He ɪ",
                    "He ɪ",
                    "He ɪ",
                    "He ɪ",
                    "H ɪ",
                ],
                "line_wavlengths": [  # 43746.46316571428, 43739.42337142857,
                    # 43695.19201,
                    # 43708.50935000001,
                    # 43739.37958,
                    43739.4215275,
                    # 43739.51833,
                    # 43744.31301,
                    # 43744.94514857143,
                    # 43745.87152666666,
                    43745.918667499995,
                    43746.04701666667,
                    43746.1930395,
                    43746.2220454,
                    43746.46316571428,
                    43764.543999999994,
                ],
            },
            # Humphrey delta
            {
                "min_lambda": 51230,
                "max_lambda": 51450,
                "main_line_name": r"Hu$\delta$",
                "line_names": [
                    "He ɪ",
                    "He ɪ",
                    "H ɪ",
                ],
                "line_wavlengths": [
                    51263.32814807571,
                    51271.5462147,
                    51286.57,
                ],
            },
        ]

        if mode == "FS":
            line_library.pop(1)
            line_library.pop(-1)
        elif mode == "MSA":
            line_library.pop(2)
            line_library.pop(1)
            line_library.pop(-1)

    else:
        raise ValueError(f"Disperser {disperser} not recognized.")

    return line_library
