#lcfit step a
# use pre-defined filter shifts per SCA from kcor

CODE="/home/rkessler/SNANA/bin/snlc_fit.exe"

$CODE \
    /project2/rkessler/SURVEYS/ROMAN/USERS/rpurohit/handshaker/tests/lcfit_base.nml \
    TEXTFILE_PREFIX out02_lcfit \
    >& out02_lcfit.log &

