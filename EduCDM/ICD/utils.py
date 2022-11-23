# coding: utf-8
import json
import logging


def output_metrics(_id, obj, wfs=None, header=None, logger=logging):
    logger.info("-------- %s: %s ----------" % (header, _id))
    logger.info("\n%s" % obj)
    if wfs is not None:  # pragma: no cover
        print(json.dumps({
            "id": _id,
            "metrics": obj
        }),
              file=wfs[header],
              flush=True)
