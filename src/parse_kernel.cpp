#include "context_model.hpp"
#include "ventus_cyclesim.h"
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using meta_data_t = ventus_kernel_metadata_t;

namespace {

uint64_t divRoundUp(uint64_t value, uint64_t divisor) {
    if (divisor == 0) {
        throw std::runtime_error("invalid zero divisor for L1 partition calculation");
    }
    return value / divisor + (value % divisor != 0);
}

uint64_t getCyclesimParam(ventus_cyclesim_param_id_t param) {
    uint64_t value = 0;
    if (ventus_cyclesim_get_param_u64(param, &value) != 0 || value == 0) {
        throw std::runtime_error("failed to query cyclesim L1 partition parameter");
    }
    return value;
}

uint64_t chooseL1dSlotCount(uint64_t maxL1dSlots, uint64_t minL1dSlots,
                            uint64_t l1dSlotGranularity, uint64_t l1dMaxSets) {
    static constexpr uint64_t kLegalL1dSlotCounts[] = {64, 128, 256, 512, 1024};
    uint64_t best = 0;
    for (uint64_t candidate : kLegalL1dSlotCounts) {
        if (candidate < minL1dSlots || candidate > maxL1dSlots) {
            continue;
        }
        if (candidate % l1dSlotGranularity != 0) {
            continue;
        }
        if (candidate / l1dSlotGranularity > l1dMaxSets) {
            continue;
        }
        best = std::max(best, candidate);
    }
    return best;
}

void assignL1PartitionMetadata(meta_data_t& metadata) {
    const uint64_t slotBytes =
        getCyclesimParam(VENTUS_CYCLESIM_PARAM_L1_PARTITION_SLOT_BYTES);
    const uint64_t totalSlots =
        getCyclesimParam(VENTUS_CYCLESIM_PARAM_L1_PARTITION_SLOT_COUNT);
    const uint64_t minL1dSlots =
        getCyclesimParam(VENTUS_CYCLESIM_PARAM_L1_MIN_L1D_SLOTS);
    const uint64_t l1dSlotGranularity =
        getCyclesimParam(VENTUS_CYCLESIM_PARAM_L1D_SLOT_GRANULARITY);
    const uint64_t l1dMaxSets =
        getCyclesimParam(VENTUS_CYCLESIM_PARAM_L1D_MAX_SETS);
    const uint64_t maxWgSlotPerSm =
        getCyclesimParam(VENTUS_CYCLESIM_PARAM_MAX_CTA_PER_SM);
    const uint64_t totalWarpsPerSm =
        getCyclesimParam(VENTUS_CYCLESIM_PARAM_NUM_WARP_PER_SM);
    const uint64_t totalSgpr =
        getCyclesimParam(VENTUS_CYCLESIM_PARAM_TOTAL_SGPR);
    const uint64_t totalVgpr =
        getCyclesimParam(VENTUS_CYCLESIM_PARAM_TOTAL_VGPR);

    metadata.ldsSlotCountPerWg = divRoundUp(metadata.ldsSize, slotBytes);
    if (metadata.ldsSlotCountPerWg > totalSlots - minL1dSlots) {
        throw std::runtime_error("kernel LDS exceeds unified L1 SMEM capacity");
    }

    uint64_t residentWgPerSm = maxWgSlotPerSm;
    if (metadata.wg_size != 0) {
        residentWgPerSm = std::min(residentWgPerSm, totalWarpsPerSm / metadata.wg_size);
    }

    const uint64_t sgprPerWg = metadata.wg_size * metadata.sgprUsage;
    const uint64_t vgprPerWg = metadata.wg_size * metadata.vgprUsage;
    if (sgprPerWg != 0) {
        residentWgPerSm = std::min(residentWgPerSm, totalSgpr / sgprPerWg);
    }
    if (vgprPerWg != 0) {
        residentWgPerSm = std::min(residentWgPerSm, totalVgpr / vgprPerWg);
    }

    while (residentWgPerSm > 0) {
        const uint64_t requiredSmemSlots =
            divRoundUp(residentWgPerSm * metadata.ldsSize, slotBytes);
        if (requiredSmemSlots <= totalSlots) {
            const uint64_t l1dSlots =
                chooseL1dSlotCount(totalSlots - requiredSmemSlots, minL1dSlots,
                                   l1dSlotGranularity, l1dMaxSets);
            if (l1dSlots != 0) {
                metadata.smemSlotCountPerSm = totalSlots - l1dSlots;
                return;
            }
        }
        residentWgPerSm--;
    }

    throw std::runtime_error("failed to choose unified L1 partition for kernel metadata");
}

} // namespace

// Helpers
bool isHexCharacter(char c) {
    return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f');
}

int charToHex(char c) {
    if (c >= '0' && c <= '9')
        return c - '0';
    else if (c >= 'A' && c <= 'F')
        return c - 'A' + 10;
    else if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
    else
        return -1; // Invalid character
}

// convert raw metadata buffer into struct meta_data_t
void assignMetadata(const std::vector<uint64_t>& rawdata, meta_data_t& metadata) {
    int index = 0;

    metadata.startaddr = rawdata[index++];

    metadata.kernel_id = rawdata[index++];

    for (int i = 0; i < 3; i++) {
        metadata.kernel_size[i] = rawdata[index++];
    }

    metadata.wf_size = rawdata[index++];
    metadata.wg_size = rawdata[index++];
    metadata.metaDataBaseAddr = rawdata[index++];
    metadata.ldsSize = rawdata[index++];
    metadata.pdsSize = rawdata[index++];
    metadata.sgprUsage = rawdata[index++];
    metadata.vgprUsage = rawdata[index++];
    metadata.pdsBaseAddr = rawdata[index++];
    assignL1PartitionMetadata(metadata);

    metadata.num_buffer = rawdata[index++];

    metadata.buffer_base = new uint64_t[metadata.num_buffer];
    for (int i = 0; i < metadata.num_buffer; i++) {
        metadata.buffer_base[i] = rawdata[index++];
    }

    metadata.buffer_size = new uint64_t[metadata.num_buffer];
    for (int i = 0; i < metadata.num_buffer; i++) {
        metadata.buffer_size[i] = rawdata[index++];
    }

    metadata.buffer_allocsize = new uint64_t[metadata.num_buffer];
    for (int i = 0; i < metadata.num_buffer; i++) {
        metadata.buffer_allocsize[i] = rawdata[index++];
    }
}

void readHexFile(const std::string& filename, int itemSize, std::vector<uint64_t>& items) {
    // itemSize为每个数据的比特数，这里为64
    std::ifstream file(filename);

    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    char c;
    int bits = 0;
    uint64_t value = 0;
    bool leftside = false;

    while (file.get(c)) {
        if (c == '\n') {
            if (bits != 0)
                leftside = true;
            continue;
        }

        if (!isHexCharacter(c)) {
            std::cerr << "Invalid character found: '" << c << "' in " << filename << std::endl;
            continue;
        }

        int hexValue = charToHex(c);
        if (leftside)
            value = value | ((uint64_t)hexValue << (92 - bits));
        else
            value = (value << 4) | hexValue;
        bits += 4;

        if (bits >= itemSize) {
            items.push_back(value);
            value = 0;
            bits = 0;
            leftside = false;
        }
    }

    if (bits > 0) {
        std::cerr << "Warning: Incomplete item found at the end of the file!" << std::endl;
    }

    file.close();
}

std::shared_ptr<ventus_kernel_metadata_t> parse_metadata(const std::filesystem::path& metafile) {
    std::vector<uint64_t> rawdata;
    auto metadata = std::make_shared<ventus_kernel_metadata_t>();
    readHexFile(metafile, 64, rawdata);
    assignMetadata(rawdata, *metadata);
    return metadata;
}
