from pprint import pprint

from src.interface.configuration import load_config
from src.inverse_model.protocols import inversion_protocol_factory


def main():
    db = load_config().database()

    for inv_protocol_id in range(len(db.inversion_protocols)):
        ip_schema = db.inversion_protocols[inv_protocol_id]
        pprint(dict(ip_schema))

        for lambdaa in ip_schema.lambdaas_schema.as_array():
            ip_kwargs = ip_schema.parameters(lambdaa=lambdaa)
            pprint(ip_kwargs)
            inverter = inversion_protocol_factory(option=ip_schema.type, parameters=ip_kwargs)
            pprint(inverter)
            print("Inverting the interferograms...")

        print("Computing the RMSE of the reconstruction vs reference spectra w.r.t lambdaas...")
        print("Selecting lambdaa_min, and the associated RMSE and reconstructed spectrum...")

        print()


if __name__ == "__main__":
    main()
