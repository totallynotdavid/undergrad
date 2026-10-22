"""
atmospheric.py, Sam Murphy (2016-10-26)

Atmospheric water vapour, ozone and AOT from GEE

Usage
H2O = Atmospheric.water(geom,date)
O3 = Atmospheric.ozone(geom,date)
AOT = Atmospheric.aerosol(geom,date)
"""

import ee


class Atmospheric:
    def round_date(date, xhour):
        """
        rounds a date of to the closest 'x' hours
        """
        y = date.get("year")
        m = date.get("month")
        d = date.get("day")
        H = date.get("hour")
        HH = H.divide(xhour).round().multiply(xhour)
        return date.fromYMD(y, m, d).advance(HH, "hour")

    def round_month(date):
        """
        round date to closest month
        """
        m1 = date.fromYMD(date.get("year"), date.get("month"), ee.Number(1))
        m2 = m1.advance(1, "month")

        d1 = ee.Number(date.difference(m1, "day")).abs()
        d2 = ee.Number(date.difference(m2, "day")).abs()

        return ee.Date(ee.Algorithms.If(d2.gt(d1), m1, m2))

    def water(geom, date):
        """
        Water vapour column above target at time of image aquisition.

        (Kalnay et al., 1996, The NCEP/NCAR 40-Year Reanalysis Project. Bull.
        Amer. Meteor. Soc., 77, 437-471)
        """

        centroid = geom.centroid()

        H2O_date = Atmospheric.round_date(date, 1)

        water_ic = ee.ImageCollection("ECMWF/ERA5/HOURLY").filterDate(
            H2O_date, H2O_date.advance(1, "hour")
        )

        water_img = ee.Image(water_ic.first())

        water = water_img.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=centroid
        ).get("total_column_water_vapour")

        # Google = kg/m^2, Py6S = g/cm^2.
        water_Py6S_units = ee.Number(water).divide(10)

        return water_Py6S_units

    def ozone(geom, date):
        """
        returns ozone measurement from merged TOMS/OMI dataset

        OR

        uses our fill value (which is mean value for that latlon and day-of-year)

        """

        centroid = geom.centroid()

        def ozone_measurement(centroid, O3_date):
            ozone_ic = ee.ImageCollection("TOMS/MERGED_V4").filterDate(
                O3_date, O3_date.advance(1, "month")
            )

            ozone_img = ee.Image(ozone_ic.first())

            ozone = ee.Algorithms.If(
                ozone_img,
                ozone_img.reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=centroid
                ).get("ozone"),
                ozone_fill(centroid, O3_date),
            )

            return ozone

        def ozone_fill(centroid, O3_date):
            """
            Gets our ozone fill value (i.e. mean value for that doy and latlon)

            you can see it
            1) compared to LEDAPS: https://code.earthengine.google.com/8e62a5a66e4920e701813e43c0ecb83e
            2) as a video: https://www.youtube.com/watch?v=rgqwvMRVguI&feature=youtu.be

            """

            ozone_fills = ee.ImageCollection(
                "users/samsammurphy/public/ozone_fill"
            ).toList(366)

            jan01 = ee.Date.fromYMD(O3_date.get("year"), 1, 1)
            doy_index = O3_date.difference(
                jan01, "day"
            ).toInt()  # (NB. index is one less than doy, so no need to +1)

            fill_image = ee.Image(ozone_fills.get(doy_index))

            return fill_image.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=centroid
            ).get("ozone")

        O3_date = Atmospheric.round_date(date, 24)

        TOMS_gap = ee.DateRange("1994-11-01", "1996-08-01")

        ozone = ee.Algorithms.If(
            TOMS_gap.contains(O3_date),
            ozone_fill(centroid, O3_date),
            ozone_measurement(centroid, O3_date),
        )

        # Fill spatial gaps and missing images with the monthly fallback.
        ozone = ee.Algorithms.If(ozone, ozone, ozone_fill(centroid, O3_date))

        ozone_Py6S_units = ee.Number(ozone).divide(
            1000
        )  # (i.e. Dobson units are milli-atm-cm )

        return ozone_Py6S_units

    def aerosol(geom, date):
        """
        Aerosol Optical Thickness.

        try:
          MODIS Aerosol Product (monthly)
        except:
          fill value
        """

        def aerosol_fill(date):
            """
            MODIS AOT fill value for this month (i.e. no data gaps)
            """
            return (
                ee.Image("users/samsammurphy/public/AOT_stack")
                .select([ee.String("AOT_").cat(date.format("M"))])
                .rename(["AOT_550"])
            )

        def aerosol_this_month(date):
            """
            MODIS AOT original data product for this month (i.e. some data gaps)
            """
            img = ee.Image(
                ee.ImageCollection("MODIS/061/MOD08_M3")
                .filterDate(Atmospheric.round_month(date))
                .first()
            )

            img = ee.Algorithms.If(
                img,
                img.select(["Aerosol_Optical_Depth_Land_Ocean_Mean_Mean"])
                .divide(1000)
                .rename(["AOT_550"]),
                aerosol_fill(date),
            )

            return img

        def get_AOT(AOT_band, geom):
            """
            AOT scalar value for target
            """
            return (
                ee.Image(AOT_band)
                .reduceRegion(reducer=ee.Reducer.mean(), geometry=geom.centroid())
                .get("AOT_550")
            )

        after_modis_start = date.difference(ee.Date("2000-03-01"), "month").gt(0)

        AOT_band = ee.Algorithms.If(
            after_modis_start, aerosol_this_month(date), aerosol_fill(date)
        )

        AOT = get_AOT(AOT_band, geom)

        # If the reduction returned nothing, use the fill value.
        AOT = ee.Algorithms.If(AOT, AOT, get_AOT(aerosol_fill(date), geom))

        return AOT
