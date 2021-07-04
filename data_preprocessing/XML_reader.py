import xml.etree.ElementTree as ET
import numpy as np
from . import simulation_dataset_config as cfg


def main():
    xml_file_path = cfg.simulation_raw_data_path + cfg.xml_file
    csv_file_path = cfg.simulation_raw_data_path + cfg.motion_file

    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    AuditId_array = []
    Timestamp_array = []
    Position_array = []
    Orientation_array = []
    Force_array = []

    for HapticStatus in root.findall('HapticStatus'):

        print(HapticStatus.tag, HapticStatus.attrib)
        AuditId_array.append(HapticStatus.attrib['AuditId'])
        Timestamp_array.append(HapticStatus.attrib['Timestamp'])

        for Position in HapticStatus.findall('Position'):
            Pos = np.asarray((Position[0].text, Position[1].text, Position[2].text))
            Position_array.append(Pos)
            print(Pos)

        for Orientation in HapticStatus.findall('Orientation'):
            Ori = np.asarray((Orientation[0].text, Orientation[1].text, Orientation[2].text, Orientation[3].text))
            Orientation_array.append(Ori)
            print(Ori)

        for Force in HapticStatus.findall('Force'):
            fce = np.asarray((Force[0].text, Force[1].text, Force[2].text))
            Force_array.append(fce)
            print(fce)

    AuditId_array = np.asarray(AuditId_array)
    AuditId_array = np.expand_dims(AuditId_array, axis=1)
    Timestamp_array = np.asarray(Timestamp_array)
    Timestamp_array = np.expand_dims(Timestamp_array, axis=1)
    Position_array = np.asarray(Position_array)
    Orientation_array = np.asarray(Orientation_array)
    Force_array = np.asarray(Force_array)

    # data format: AuditId, Timestamp, Position.x, Position.y, Position.z, Orientation.x, Orientation.y, Orientation.z, Orientation.w, Force.x, Force.y, Force.z
    csv_data = np.hstack((AuditId_array, Timestamp_array, Position_array, Orientation_array, Force_array))
    np.savetxt(csv_file_path, csv_data, delimiter=',', fmt='%s')
    return


if __name__ == '__main__':
    main()