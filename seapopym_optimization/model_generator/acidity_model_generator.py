"""Model generator for SeapoPym AcidityModel."""

from dataclasses import dataclass, field

from seapopym.configuration.acidity import (
    AcidityConfiguration,
    ForcingParameter,
    FunctionalGroupParameter,
    FunctionalTypeParameter,
)
from seapopym.configuration.no_transport import (
    EnvironmentParameter,
    FunctionalGroupUnit,
    KernelParameter,
    MigratoryTypeParameter,
)
from seapopym.model import AcidityModel

from seapopym_optimization.model_generator.no_transport_model_generator import NoTransportModelGenerator


@dataclass(kw_only=True)
class AcidityModelGenerator(NoTransportModelGenerator):
    forcing_parameters: ForcingParameter
    model_type: type[AcidityModel] = AcidityModel
    environment: EnvironmentParameter | None = field(default_factory=EnvironmentParameter)
    kernel: KernelParameter | None = field(default_factory=KernelParameter)

    def generate(
        self, functional_group_parameters: list[dict[str, float]], functional_group_names: list[str] | None = None
    ) -> AcidityModel:
        """
        Generate a AcidityModel with the given functional group parameters and names.

        Parameters
        ----------
        functional_group_parameters: list[dict[str, float]]
            A list of dictionaries where each dictionary contains the parameters for a functional group.
            Each dictionary should have keys corresponding to the parameter names defined in the FunctionalGroupUnit.
        functional_group_names: list[str] | None
            A list of names for the functional groups.
            If None, default names will be used (e.g., "Group_0", "Group_1", etc.).

        Returns
        -------
        AcidityModel
            A AcidityModel object containing the functional groups with their parameters.

        """

        def create_functional_group_unit(fg_num: int, fg_parameter: dict[str, float]) -> FunctionalGroupUnit:
            fg_name = f"Group_{fg_num}" if functional_group_names is None else functional_group_names[fg_num]
            return FunctionalGroupUnit(
                name=fg_name,
                energy_transfert=fg_parameter["energy_transfert"],
                migratory_type=MigratoryTypeParameter(
                    day_layer=fg_parameter["day_layer"],
                    night_layer=fg_parameter["night_layer"],
                ),
                functional_type=FunctionalTypeParameter(
                    lambda_temperature_0=fg_parameter["lambda_temperature_0"],
                    gamma_lambda_temperature=fg_parameter["gamma_lambda_temperature"],
                    tr_0=fg_parameter["tr_0"],
                    gamma_tr=fg_parameter["gamma_tr"],
                    lambda_acidity_0=fg_parameter["lambda_acidity_0"],
                    gamma_lambda_acidity=fg_parameter["gamma_lambda_acidity"],
                ),
            )

        functional_group_set = [
            create_functional_group_unit(fg_num, fg_parameter)
            for fg_num, fg_parameter in enumerate(functional_group_parameters)
        ]

        model_configuration = AcidityConfiguration(
            forcing=self.forcing_parameters,
            functional_group=FunctionalGroupParameter(functional_group=functional_group_set),
            environment=self.environment,
            kernel=self.kernel,
        )

        return AcidityModel.from_configuration(model_configuration)
