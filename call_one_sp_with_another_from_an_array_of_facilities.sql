-- =============================================================================
-- STREAMLINED BATCH PROCESSING STORED PROCEDURE - LOGIC ONLY
-- =============================================================================

CREATE OR REPLACE PROCEDURE `thcdnatestdata.staging.test_Sp_dim_financial_class_streamlined`()
BEGIN

  -- Facility codes array
  DECLARE facility_codes ARRAY<STRING> DEFAULT [
    'AHD','AHH','BAR','BMA','BMC','CCD','CHF','CSK','CSM','DEL','DES','ECH','EMC','FUH',
    'FVR','GSM','HHH','HMD','IND','LAK','LMF','LOM','MAN','MOD','MTB','NBH','NCA','NMC',
    'PBA','PBG','PLA','PMC','PRV','PVA','RHB','SES','SFH','SIE','SLH','SMH','SPW','SRM',
    'SVH','SVM','TWI','VBA','VBC','WBO','WHF','WVH'
  ];

  -- Variables for processing
  DECLARE i INT64 DEFAULT 0;
  DECLARE OUT_PARAM INT64;
  DECLARE inparam_facility_cd STRING;
  DECLARE total_facilities INT64;
  DECLARE retry_count INT64;
  DECLARE max_retries INT64 DEFAULT 2;

  -- Calculate total facilities
  SET total_facilities = ARRAY_LENGTH(facility_codes);

  -- Loop through each facility code
  WHILE i < total_facilities DO
    SET inparam_facility_cd = facility_codes[OFFSET(i)];
    SET retry_count = 0;

    -- Retry loop for each facility
    WHILE retry_count <= max_retries DO
      BEGIN
        -- Call the optimized stored procedure
        CALL `thcdnapreproddata.idm.sp_daac_to_idm_dim_financial_class_test_20082025`(
          inparam_facility_cd, OUT_PARAM
        );

        -- If successful, exit retry loop
        IF OUT_PARAM = 1 THEN
          SET retry_count = max_retries + 1; -- Exit retry loop
        ELSE
          SET retry_count = max_retries + 1; -- Exit retry loop even on failure
        END IF;

      EXCEPTION WHEN ERROR THEN
        SET retry_count = retry_count + 1;
        
        IF retry_count > max_retries THEN
          -- Max retries reached, continue to next facility
        END IF;
      END;
    END WHILE;

    SET i = i + 1;
  END WHILE;

END;
