from torch import nn

from core.cost.ADC import ADC
from core.cost.DAC import DAC

__all__ = ["CostPredictor"]


class CostPredictor(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        cost_name: str = "area",
        work_freq: int = 5,
        work_prec: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = model
        self.cost_name = cost_name
        self.work_freq = work_freq
        self.work_prec = work_prec

    def get_num_cr(self, gene_cr):  # get the number of crossings
        nums = 0
        n = len(gene_cr)
        gene_cr_copy = gene_cr.copy()
        for i in range(n):
            for j in range(0, n - i - 1):
                if gene_cr_copy[j] > gene_cr_copy[j + 1]:
                    gene_cr_copy[j], gene_cr_copy[j + 1] = (
                        gene_cr_copy[j + 1],
                        gene_cr_copy[j],
                    )
                    nums += 1
        return nums

    def counter_func(self, a, k):  # count the number of columns used in cr array
        b = 1
        decrement = k - 1
        while a > decrement:
            b += 1
            a -= decrement
            decrement -= 1
        return b

    def _evaluate_area(self, arch_sol):  # changed to total area
        if isinstance(arch_sol, str):
            arch_sol = eval(arch_sol)

        # get all the device parameters: width, height, spacing
        k = self.model.super_layer_config["n_waveguides"]

        spacing = self.model.super_layer_config["device_cost"][
            "spacing"
        ]  # change to 250um

        h_spacing = self.model.super_layer_config["device_cost"]["h_spacing"]

        ps_width = self.model.super_layer_config["device_cost"]["ps_cost"]["width"]

        ps_height = self.model.super_layer_config["device_cost"]["ps_cost"]["height"]

        cr_width = self.model.super_layer_config["device_cost"]["cr_cost"]["width"]

        cr_height = self.model.super_layer_config["device_cost"]["cr_cost"]["height"]

        cr_spacing = self.model.super_layer_config["device_cost"]["cr_cost"][
            "cr_spacing"
        ]

        # input modulator
        modulator_width = self.model.super_layer_config["device_cost"][
            "modulator_cost"
        ]["width"]

        modulator_length = self.model.super_layer_config["device_cost"][
            "modulator_cost"
        ]["length"]

        # Photodetector
        photodetector_width = self.model.super_layer_config["device_cost"][
            "photodetector_cost"
        ]["width"]

        photodetector_length = self.model.super_layer_config["device_cost"][
            "photodetector_cost"
        ]["length"]

        # sigma
        y_branch_width = self.model.super_layer_config["device_cost"]["y_branch_cost"][
            "width"
        ]  # horizontal

        y_branch_length = self.model.super_layer_config["device_cost"]["y_branch_cost"][
            "length"
        ]  # vertical

        TIA_area = (
            k * self.model.super_layer_config["device_cost"]["TIA_cost"]["area"]
        )  # TIA

        adc_area = self.model.super_layer_config["device_cost"]["adc_cost"][1][
            "area"
        ]  # ADC shared

        dac_area = self.model.super_layer_config["device_cost"]["dac_cost"][1][
            "area"
        ]  # DAC *#ps

        ps_area, dc_area, cr_area, dac_area_total = [0, 0, 0, 0]

        def _evaluate_area_ps(block):
            ps_area = 0
            ps_area = ps_area + (ps_width + h_spacing) * (
                ps_height + ((k - 1) * spacing)
            )
            return ps_area

        def _evaluate_area_dc(block):  # 10/22: updated, support arbitrary port number
            dc_area = 0
            max_num = max(
                block[1][1]
            )  # get the port number of the largest DC in the array. eg.[1,4,2,1], max_num = 4
            if max_num > 1:
                key = f"dc{max_num}_cost"  # get the key to device cost. eg. max_num = 4, key = "dc4_cost"
                dc_width = self.model.super_layer_config["device_cost"][key][
                    "width"
                ]  # get the width of the largest DC
                dc_length = self.model.super_layer_config["device_cost"][key]["length"]
                dc_area = dc_area + (dc_width + h_spacing) * (
                    dc_length + (k - 1) * spacing
                )  # use the width as the DC array width
            return dc_area

        def _evaluate_area_cr(block):
            cr_area = 0

            nums = self.get_num_cr(block[1][1])
            num_row = k - 1 if nums > k - 1 else nums  # count the number of rows used

            num_col = self.counter_func(nums, k)
            cr_area = (
                cr_area
                + ((k - 1) * spacing) * h_spacing
                + (cr_height + (num_row - 1) * cr_spacing)
                * (cr_width + (num_col - 1) * cr_spacing)
            )

            return cr_area

        for i in range(len(arch_sol)):
            if arch_sol[i][1][0] == 1:
                if i % 2 == 0:
                    ps_area = ps_area + _evaluate_area_ps(
                        arch_sol[i]
                    )  # Add one full column of PS if block is valid
                    dac_area_total = dac_area_total + k * dac_area
                    dc_area = dc_area + _evaluate_area_dc(arch_sol[i])
                else:
                    cr_area = cr_area + _evaluate_area_cr(arch_sol[i])

        unitary_area = ps_area + dc_area + cr_area

        ## 12/12 Update: Area for sigma: two y_branch and two phase shifters
        sigma_area = (2 * y_branch_width + ps_width) * (
            (k - 1) * spacing + ps_height + y_branch_length
        )

        photodetector_area = k * (photodetector_width * photodetector_length)
        modulator_area = k * (modulator_length * modulator_width)

        # print("Unitary area:", unitary_area)
        # print("TIA area:", TIA_area)
        # print("adc area:", adc_area)
        # print("dac area:", dac_area_total)
        # print("sigma area:", sigma_area)
        # print("photodetector area:", photodetector_area)
        # print("modulator area:", modulator_area)

        cost_area = (
            unitary_area  # U and V
            + TIA_area
            + adc_area
            + dac_area_total
            + sigma_area  # Sigma
            + photodetector_area
            + modulator_area  # Input Modulator
        )
        return cost_area  # Unit: um^2

    def _evaluate_power(self, arch_sol):  ## changed to total power
        # Total power(unit mW)
        # input: laser, input modulator, DAC
        # output: photodetector, TIA, ADC
        # computation: sigma(insertion loss on Y branch/PS, static power of PS)

        if isinstance(arch_sol, str):
            arch_sol = eval(arch_sol)

        k = self.model.super_layer_config["n_waveguides"]

        def cal_insertion_loss(
            arch_sol,
        ):  # calculate total insertion loss on the longest path
            # insertion loss of one phase shifter
            ps_loss = self.model.super_layer_config["device_cost"]["ps_cost"][
                "insertion_loss"
            ]

            # insertion loss of one Y-branch
            y_branch_loss = self.model.super_layer_config["device_cost"][
                "y_branch_cost"
            ]["insertion_loss"]

            # insertion loss of one waveguide crossing
            cr_loss = self.model.super_layer_config["device_cost"]["cr_cost"][
                "insertion_loss"
            ]

            # insertion loss of input modulator
            modulator_loss = self.model.super_layer_config["device_cost"][
                "modulator_cost"
            ]["insertion_loss"]

            # insertion loss of sigma
            # The path for one port: two Y branch and two PS
            sigma_loss = 2 * y_branch_loss + ps_loss

            total_loss = 0

            def get_cr_num(
                block,
            ):  # this function is to get the total number of crossings for each CR array
                index_list = list(range(k))
                result_list = [abs(a - b) for a, b in zip(block[1][1], index_list)]
                return max(result_list)

            def get_insertion_loss_dc(
                block,
            ):  # this function is to get the dc_loss for each DC array
                dc_loss = 0
                max_num = max(
                    block[1][1]
                )  # get the port number of the largest DC in the array. eg.[1,4,2,1], max_num = 4
                if (
                    max_num > 1
                ):  # If no DC exists, dc_loss = 0, otherwise get the dc_loss from config file
                    key = f"dc{max_num}_cost"  # get the key to device cost. eg. max_num = 4, key = "dc4_cost"
                    dc_loss = self.model.super_layer_config["device_cost"][key][
                        "insertion_loss"
                    ]  # get the insertion_loss of the largest DC
                return dc_loss  # assume the largest DC has the largest dc_loss, use it to represent the dc_loss of this DC array

            for i in range(len(arch_sol)):
                if arch_sol[i][1][0] == 1:
                    if i % 2 == 0:
                        dc_loss = get_insertion_loss_dc(arch_sol[i])
                        total_loss = total_loss + ps_loss + dc_loss
                    else:
                        num_cr = get_cr_num(arch_sol[i])
                        total_loss = total_loss + num_cr * cr_loss

            total_loss = total_loss + modulator_loss + sigma_loss
            return total_loss  # total insertion loss(dB)

        def cal_laser_power(arch_sol):  # this function is to calculate laser power
            pd_sensitivity = self.model.super_layer_config["device_cost"][
                "photodetector_cost"
            ]["sensitivity"]
            resolution = self.model.super_layer_config["device_cost"]["resolution"]
            laser_wall_plug_eff = self.model.super_layer_config["device_cost"][
                "laser_wall_plug_eff"
            ]

            p_laser_dBm = pd_sensitivity + cal_insertion_loss(
                arch_sol
            )  # laser power: unit dBm
            p_laser = (
                10 ** (p_laser_dBm / 10) / laser_wall_plug_eff * (2**resolution)
            )  # convert dBm to mW
            return p_laser  # laser power: unit mW

        def get_num_ps_layers(
            arch_sol,
        ):  # this function is to get the number of valid PS layers
            cnt = 0
            for i in range(len(arch_sol)):
                if arch_sol[i][1][0] == 1:
                    if i % 2 == 0:
                        cnt = cnt + 1
            return cnt

        laser_power = cal_laser_power(arch_sol)

        # Add ps static power(currently set to 0)
        ps_static_power = self.model.super_layer_config["device_cost"]["ps_cost"][
            "static_power"
        ]
        # print("ps_static_power:", ps_static_power)

        ps_num = k * get_num_ps_layers(arch_sol)

        dac_device = DAC(choice=1)
        dac_device.set_DAC_work_freq(self.work_freq)
        dac_device.set_DAC_work_prec(self.work_prec)
        dac_device.cal_DAC_param()
        dac_power = dac_device.DAC_power
        dac_num = k * get_num_ps_layers(arch_sol)

        adc_device = ADC(choice=1)
        adc_device.set_ADC_work_freq(self.work_freq)  # set work freq and bit width
        adc_device.set_ADC_work_prec(self.work_prec)  # set bit width
        adc_device.cal_ADC_param()  # convert power to desired freq and bit width
        adc_power = adc_device.ADC_power

        modulator_power = self.model.super_layer_config["device_cost"][
            "modulator_cost"
        ]["static_power"]  # power of input modulator

        photodetector_power = self.model.super_layer_config["device_cost"][
            "photodetector_cost"
        ]["power"]  # power of photodetector

        TIA_power = self.model.super_layer_config["device_cost"]["TIA_cost"][
            "power"
        ]  # power of TIA

        # static power consumption of sigma
        sigma_static_power = 2 * k * ps_static_power

        total_power = (
            laser_power
            + k
            * (modulator_power + photodetector_power + TIA_power + sigma_static_power)
            + adc_power
            + (dac_power * dac_num)
            + (ps_num * ps_static_power)
        )  # changed ADC/DAC power
        return total_power
        # add phase shifter static power(currently assume 0)

    def _evaluate_latency(self, arch_sol):  # change to total latency
        if isinstance(arch_sol, str):
            arch_sol = eval(arch_sol)

        k = self.model.super_layer_config["n_waveguides"]
        n_group = self.model.super_layer_config["device_cost"]["n_group"]
        spacing = self.model.super_layer_config["device_cost"]["spacing"]
        h_spacing = self.model.super_layer_config["device_cost"]["h_spacing"]

        ps_width = self.model.super_layer_config["device_cost"]["ps_cost"]["width"]

        cr_width = self.model.super_layer_config["device_cost"]["cr_cost"]["width"]
        cr_height = self.model.super_layer_config["device_cost"]["cr_cost"]["height"]
        cr_spacing = self.model.super_layer_config["device_cost"]["cr_cost"][
            "cr_spacing"
        ]

        modulator_width = self.model.super_layer_config["device_cost"][
            "modulator_cost"
        ]["width"]  # Input Modulator width

        y_branch_width = self.model.super_layer_config["device_cost"]["y_branch_cost"][
            "width"
        ]  # Y Branch width(horizontal)

        adc_sample_rate = self.model.super_layer_config["device_cost"]["adc_cost"][1][
            "sample_rate"
        ]  # unit GHz

        ## 12/12 update: adc latency is not added to the total latency.
        ## The work frequency of ADC cannot exceed sample rate

        dac_latency = self.model.super_layer_config["device_cost"]["dac_cost"][1][
            "latency"
        ]  # DAC get latency directly 10ps

        photodetector_latency = self.model.super_layer_config["device_cost"][
            "photodetector_cost"
        ]["latency"]  # 10ps

        def get_width_dc(block):  # this function is to get the width for each DC array
            dc_width = 0
            max_num = max(
                block[1][1]
            )  # get the port number of the largest DC in the array. eg.[1,4,2,1], max_num = 4
            if (
                max_num > 1
            ):  # If no DC exists, dc_loss = 0, otherwise get the dc_loss from config file
                key = f"dc{max_num}_cost"  # get the key to device cost. eg. max_num = 4, key = "dc4_cost"
                dc_width = self.model.super_layer_config["device_cost"][key][
                    "width"
                ]  # get the insertion_loss of the largest DC
            return dc_width  # assume the largest DC has the largest dc_loss, use it to represent the dc_loss of this DC array

        ps_path, dc_path, cr_path = [0, 0, 0]

        num_active_block = 0

        for i in range(len(arch_sol)):
            if arch_sol[i][1][0] == 1:
                if i % 2 == 0:
                    num_active_block = num_active_block + 1
                    ps_path = ps_path + (ps_width + h_spacing)
                    dc_path = (
                        get_width_dc(arch_sol[i]) + h_spacing
                    )  # modidy dc_path, get from correct dc_cost based on the maximum port number in the array
                else:
                    # num_cr = get_cr_num(arch_sol[i])
                    nums = self.get_num_cr(arch_sol[i][1][1])
                    num_row = (
                        k - 1 if nums > k - 1 else nums
                    )  # count the number of rows used
                    num_col = self.counter_func(nums, k)
                    if num_row == 0:
                        cr_path = cr_path + h_spacing
                    if num_row == 1:
                        cr_path = cr_path + cr_width + h_spacing
                    else:
                        cr_path = (
                            cr_path
                            + (num_row - 1) * cr_height
                            + (num_row - 1) * cr_spacing
                            + ((num_col - 1) * cr_width)
                            + ((num_col - 1) * cr_spacing)
                            + h_spacing
                        )  ##cr_spacing

        unitary_path = ps_path + dc_path + cr_path  # total path length, unit um
        # print("Unitary path:", unitary_path)

        sigma_path = 2 * y_branch_width + ps_width
        # print("Sigma path:", sigma_path)

        total_path = modulator_width + unitary_path + sigma_path

        c0 = 3e8  # speed of light, unit m/s
        latency = total_path / (c0 / n_group) * 1e6  # latency, unit ps

        latency = (
            latency
            + dac_latency  # 10ps
            + photodetector_latency  # 10ps
        )

        # print("Latency:", latency)
        min_latency = (
            1e3 / adc_sample_rate
        )  # minimum latency constrained by adc sample rate
        return latency if latency > min_latency else min_latency

    def forward(self, arch_sol, cost_name: str = "area.power.latency"):
        cost = {}
        if "area" in cost_name:
            cost["area"] = self._evaluate_area(arch_sol)
        if "power" in cost_name:
            cost["power"] = self._evaluate_power(arch_sol)
        if "latency" in cost_name:
            cost["latency"] = self._evaluate_latency(arch_sol)
        # else:
        #     raise NotImplementedError
        return cost
