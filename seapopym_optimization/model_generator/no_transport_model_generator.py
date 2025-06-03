"""Model generator for SeapoPym NoTransportModel."""

from dataclasses import dataclass, field

from seapopym.configuration.no_transport import (
    EnvironmentParameter,
    ForcingParameter,
    FunctionalGroupParameter,
    FunctionalGroupUnit,
    FunctionalTypeParameter,
    KernelParameter,
    MigratoryTypeParameter,
    NoTransportConfiguration,
)
from seapopym.model import NoTransportModel

from seapopym_optimization.model_generator.base_model_generator import AbstractModelGenerator


@dataclass(kw_only=True)
class NoTransportModelGenerator(AbstractModelGenerator):
    """
    Model generator for SeapoPym NoTransportModel.
    This class is responsible for generating a NoTransportModel with the specified functional group parameters and
    names. It uses the FunctionalGroupUnit to create functional groups based on the provided parameters.

    Attributes
    ----------
    model_type: type[NoTransportModel]
        The type of model to generate, which is NoTransportModel in this case.
    forcing_parameters: ForcingParameter
        The parameters related to the forcing conditions of the model.
    environment: EnvironmentParameter
        The parameters related to the environment in which the model operates.
    kernel: KernelParameter
        The parameters related to the kernel of the model, which may include spatial or temporal aspects.

    Methods
    -------
    generate(functional_group_parameters: list[dict[str, float]], functional_group_names: list[str] | None) -> NoTransportModel
        Generate a NoTransportModel with the given functional group parameters and names.
        Each functional group is created using the FunctionalGroupUnit with the specified parameters.

    """

    forcing_parameters: ForcingParameter
    model_type: type[NoTransportModel] = NoTransportModel
    environment: EnvironmentParameter | None = field(default_factory=EnvironmentParameter)
    kernel: KernelParameter | None = field(default_factory=KernelParameter)

    def generate(
        self, functional_group_parameters: list[dict[str, float]], functional_group_names: list[str] | None = None
    ) -> NoTransportModel:
        """
        Generate a NoTransportModel with the given functional group parameters and names.

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
        NoTransportModel
            A NoTransportModel object containing the functional groups with their parameters.

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
                ),
            )

        functional_group_set = [
            create_functional_group_unit(fg_num, fg_parameter)
            for fg_num, fg_parameter in enumerate(functional_group_parameters)
        ]

        model_configuration = NoTransportConfiguration(
            forcing=self.forcing_parameters,
            functional_group=FunctionalGroupParameter(functional_group=functional_group_set),
            environment=self.environment,
            kernel=self.kernel,
        )

        return NoTransportModel.from_configuration(model_configuration)
