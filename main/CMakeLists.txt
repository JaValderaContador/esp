
#
# Main component of TF Micro project 'micro_speech'.
#

idf_component_register(
    SRCS main.cc main_functions.cc constants.cc output_handler.cc images model_int8.cc
    PRIV_REQUIRES spi_flash
    INCLUDE_DIRS "")
