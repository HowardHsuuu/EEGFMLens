"""List the model connection contracts without downloading anything."""

from eeglens import supported_models

if __name__ == "__main__":
    for spec in supported_models():
        variants = ", ".join(spec.variants)
        options = []
        for integration in spec.integrations:
            required = ", ".join(integration.required_options) or "none"
            optional = ", ".join(integration.optional_options) or "none"
            options.append(f"{integration.variant}: required={required}; optional={optional}")
        print(f"{spec.display_name} [{spec.family}] variants={variants}")
        print(f"  input: {spec.input_contract}")
        print(f"  selection: {spec.physical_selection}")
        print(f"  options: {' | '.join(options)}")
