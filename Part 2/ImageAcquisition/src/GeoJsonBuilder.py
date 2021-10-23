
"""
A script used to create polygon GeoJson objects that can be viewed
to check each bbox is what it is expected to be
"""

def main():
    all_features = "{\n\"type\": \"FeatureCollection\",\n\"features\": [\n"

    for bbox_key in bbox_dict:
        bbox = bbox_dict[bbox_key]

        bbox_coords = bbox.split(',')

        cur_feature = feature_template
        cur_feature = cur_feature + "\"" + bbox_key + "\""
        cur_feature += feature_template_start_2
        cur_feature += '[' + bbox_coords[0] + ',' + bbox_coords[1] +'],\n'
        cur_feature += '[' + bbox_coords[2] + ',' + bbox_coords[1] +'],\n'
        cur_feature += '[' + bbox_coords[2] + ',' + bbox_coords[3] +'],\n'
        cur_feature += '[' + bbox_coords[0] + ',' + bbox_coords[3] +'],\n'
        cur_feature += '[' + bbox_coords[0] + ',' + bbox_coords[1] +']\n'
        cur_feature += feature_template_end

        all_features += cur_feature + ","

    all_features = all_features[:-1]+ "]}"

    print (all_features)



feature_template = "{\n \"type\": \"Feature\",\n\"properties\": {\n \"name\": "
feature_template_start_2 = "},\n\"geometry\": {\n\"type\": \"Polygon\",\n\"coordinates\": [\n[\n"
feature_template_end = "]\n]\n}\n}"

AmaQuad1_1 = "-64.0283203125, 0.5273363048115169, -59.853515625, 3.908098881894123"
AmaQuad1_2 = "-68.92822265625, 0.4833927027896987, -66.9287109375, 1.845383988573187"
AmaQuad1_3 = "-68.8623046875, -4.36832042087623, -58.7, 0.3076157096439005"

AmaQuad2_1 = "-58.447265625, -4.674979814820243, -50.9765625, 1.098565496040652"
AmaQuad2_2 = "-50.8447265625, -4.740675384778361, -46.845703125, -1.669685500986571"
AmaQuad2_3 = "-46.7578125, -3.9519408561575817, -45.3076171875, -1.7575368113083125"

AmaQuad3_1 = "-70.2685546875, -11.005904459659451, -60.380859375, -4.302591077119676"
AmaQuad3_2 = "-72.50976562499999, -9.579084335882534, -70.3125, -4.784468966579362"

AmaQuad4_1 = "-60.3369140625, -10.790140750321738, -53.96484375, -4.521666342614791"
AmaQuad4_2 = "-53.78906249999999, -7.798078531355303, -49.7900390625, -4.696879026871413"
AmaQuad4_3 = "-53.6572265625, -12.554563528593656, -52.82226562499999, -7.972197714386866"
AmaQuad4_4 = "-52.470703125, -9.88227549342994, -51.328125, -8.059229627200192"

CatQuad1_1 = "-42.23296503629097,-8.111804629213445, -39.17876581754097,-3.0618983616936823"

CatQuad2_1 = "-39.13330078125,-8.515835561202218, -35.595703125, -5.266007882805485"
CatQuad2_2 = "-35.57373046875, -8.515835561202218, -35.15625, -6.140554782450295"

CatQuad3_1 = " -40.078125, -10.09867012060338, -36.298828125, -8.537565350804018"
CatQuad3_2 = "-40.05615234375, -12.490213662533295, -38.3642578125, -10.163560279490476"
CatQuad3_3 = "-40.0341796875, -14.75363533154043, -39.5068359375, -12.490213662533295"
CatQuad3_4 = "-38.29833984375, -11.587669416896203, -37.8369140625, -10.18518740926906"

CatQuad4_1 = "-42.0556640625, -14.77488250651626, -40.0341796875, -8.146242825034385"
CatQuad4_2 = "-42.12158203124999,-15.866241564066616, -40.693359375, -14.84923123791421"

CerQuad1_1 = "-46.6259765625, -16.34122561920748, -42.84667968749999, -8.581021215641842"
CerQuad1_2 = "-45.6591796875, -8.363692651835823, -42.890625, -6.533645130567532"
CerQuad1_3 = "-44.384765625, -6.468151012664202, -42.91259765625, -5.156598738411155"

CerQuad2_1 = "-55.06347656249999, -17.056784609942543, -46.71386718749999, -14.349547837185362"
CerQuad2_2 = "-50.6689453125, -14.221788628397585, -46.6259765625,-11.73830237143684"
CerQuad2_3 = "-48.8232421875, -11.566143767762844, -46.669921875, -10.487811882056683"
CerQuad2_4 = "-47.197265625, -10.31491928581316,-46.58203125, -8.49410453755187"

CerQuad3_1 = "-49.482421875, -24.766784522874428, -48.2958984375, -17.182779056431826"
CerQuad3_2 = "-48.2080078125, -23.52370005882413, -46.845703125, -17.014767530557823"
CerQuad3_3 = "-46.669921875, -21.555284406923178, -45.3076171875, -16.425547506916725"
CerQuad3_4 = "-45.24169921875,-20.529933125170764, -43.81347656249999,-16.383391123608387"
CerQuad3_5 = "-43.70361328125, -19.476950206488414, -42.86865234375, -16.34122561920748"

CerQuad4_1 = "-54.9755859375, -22.75592068148639, -49.7900390625, -17.09879223767869"
CerQuad4_2 = "-50.625, -24.846565348219734, -49.658203125, -22.958393318086337"
CerQuad4_3 = "-57.52441406249999, -21.657428197370642, -55.1953125,-19.26966529650232"

single_quad = "-64.16015625000001, 3.8642546157213955, -63.98437500000001, 4.039617826768424"
bbox_dict = {
    "single_quad" : single_quad,
    "AmaQuad1_1" : AmaQuad1_1,
}
bbox_dict2 = {
    "AmaQuad1_1" : AmaQuad1_1,
    "AmaQuad1_2" : AmaQuad1_2,
    "AmaQuad1_3" : AmaQuad1_3,
    "AmaQuad2_1" : AmaQuad2_1,
    "AmaQuad2_2" : AmaQuad2_2,
    "AmaQuad2_3" : AmaQuad2_3,
    "AmaQuad3_1" : AmaQuad3_1,
    "AmaQuad3_2" : AmaQuad3_2,
    "AmaQuad4_1" : AmaQuad4_1,
    "AmaQuad4_2" : AmaQuad4_2,
    "AmaQuad4_3" : AmaQuad4_3,
    "AmaQuad4_4" : AmaQuad4_4,

    "CatQuad1_1" : CatQuad1_1,
    "CatQuad2_1" : CatQuad2_1,
    "CatQuad2_2" : CatQuad2_2,
    "CatQuad3_1" : CatQuad3_1,
    "CatQuad3_2" : CatQuad3_2,
    "CatQuad3_3" : CatQuad3_3,
    "CatQuad3_4" : CatQuad3_4,
    "CatQuad4_1" : CatQuad4_1,
    "CatQuad4_2" : CatQuad4_2,

    "CerQuad1_1" : CerQuad1_1,
    "CerQuad1_2" : CerQuad1_2,
    "CerQuad1_3" : CerQuad1_3,
    "CerQuad2_1" : CerQuad2_1,
    "CerQuad2_2" : CerQuad2_2,
    "CerQuad2_3" : CerQuad2_3,
    "CerQuad2_4" : CerQuad2_4,
    "CerQuad3_1" : CerQuad3_1,
    "CerQuad3_2" : CerQuad3_2,
    "CerQuad3_3" : CerQuad3_3,
    "CerQuad3_4" : CerQuad3_4,
    "CerQuad3_5" : CerQuad3_5,
    "CerQuad4_1" : CerQuad4_1,
    "CerQuad4_2" : CerQuad4_2,
    "CerQuad4_3" : CerQuad4_3

}

if __name__ == "__main__":
    main()
