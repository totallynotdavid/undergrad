import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    from Py6S import AeroProfile, SixS, SixSHelpers, Wavelength

    s = SixS()
    s.wavelength = Wavelength(0.675)
    s.aero_profile = AeroProfile.PredefinedType(AeroProfile.Maritime)
    s.run()
    (
        print,
        s.outputs.pixel_reflectance,
        s.outputs.pixel_radiance,
        s.outputs.direct_solar_irradiance,
    )
    wavelengths, results = SixSHelpers.Wavelengths.run_vnir(
        s, output_name="pixel_radiance"
    )
    SixSHelpers.Wavelengths.plot_wavelengths(
        wavelengths, results, "Pixel radiance ($W/m^2$)"
    )
    return


if __name__ == "__main__":
    app.run()
