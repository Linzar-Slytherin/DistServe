import dataclasses
from typing import List
import marshal

@dataclasses.dataclass
class TestRequest:
    """
    TestRequest: A request for testing the server's performance
    """
    
    prompt: str
    prompt_len: int
    output_len: int
    
@dataclasses.dataclass
class Dataset:
    """
    Dataset: A dataset for testing the server's performance
    """
 
    dataset_name: str	# "sharegpt" / "alpaca" / ...
    reqs: List[TestRequest]
    
    def dump(self, output_path: str):
        marshal.dump({
            "dataset_name": self.dataset_name,
            "reqs": [(req.prompt, req.prompt_len, req.output_len) for req in self.reqs]
        }, open(output_path, "wb"))
    
    @staticmethod
    def load(input_path: str):
        loaded_data = marshal.load(open(input_path, "rb"))
        return Dataset(
            loaded_data["dataset_name"],
            [TestRequest(req[0], req[1], req[2]) for req in loaded_data["reqs"]]
        )
        
import dataclasses
import numpy as np
from typing import List
import json

from enum import Enum
import dataclasses
import time
import json
from typing import List

"""
This file (lifetime.py) contains utilities for lifetime monitoring of a request
Used for micro-benchmark in the paper "disaggregate"

A request's lifetime looks like this:

Issued
|
| Context Queuing
|
Context Begin
|
| Context-ing
|
Context End
|
| Waiting in the bridge queue
|
Migration Begin
|
| Migrating
|
Migration End
|
| Decoding Queuing
|
Decoding Begin
|
| Decoding
|
Decoding End
"""

class LifetimeEventType(Enum):
    """
    The type of an event in a request's lifetime
    """
    Issued = "issued"
    ContextBegin = "context_begin"
    ContextEnd = "context_end"
    MigrationBegin = "migration_begin"
    MigrationEnd = "migration_end"
    DecodingBegin = "decoding_begin"
    DecodingEnd = "decoding_end"

    def __str__(self) -> str:
        return self.value

class LifetimeEvent(json.JSONEncoder):
    """
    An event in a request's lifetime
    Contains a timestamp and a type
    """
    
    def __init__(self, event_type: LifetimeEventType, timestamp: float = None):
        if timestamp is None:
            timestamp = time.time()
        self.event_type = event_type
        self.timestamp = timestamp
        
def json_encode_lifetime_events(events: List[LifetimeEvent]) -> List[dict]:
    return [
        {
            "timestamp": event.timestamp,
            "event_type": str(event.event_type)
        } 
        for event in events
    ]

def json_decode_lifetime_events(json_list: List[dict]) -> List[LifetimeEvent]:
    return [
        LifetimeEvent(LifetimeEventType(event["event_type"]), event["timestamp"])
        for event in json_list
    ]

class RequestResult:
    """
    A class for storing the results of a single request
    """
    
    def __init__(
        self,
        prompt_len: int,
        output_len: int,
        start_time: float,
        end_time: float,
        token_timestamps: List[float],
        lifetime_events: List[LifetimeEvent] = None
    ):
        self.prompt_len = prompt_len
        self.output_len = output_len
        self.start_time = start_time
        self.end_time = end_time
        self.token_timestamps = token_timestamps
        self.lifecycle_events = lifetime_events
        
        self.latency = end_time - start_time
        self.ftl = token_timestamps[0] - start_time
        self.tpot = 0 if output_len == 1 else (token_timestamps[-1] - token_timestamps[0]) / (output_len-1)

def read_request_results(path: str) -> List[RequestResult]:
    with open(path, "r") as f:
        request_results: List[RequestResult] = [
            RequestResult(
                item["prompt_len"],
                item["output_len"],
                item["start_time"],
                item["end_time"],
                item["token_timestamps"],
                json_decode_lifetime_events(item["lifecycle_events"]) if item.get("lifecycle_events", None) is not None else None
            )
            for item in json.load(f)
        ]
    return request_results

def count_valid_results(request_results: list[RequestResult], ftl: float, tpot: float) -> int:
    """
    count_valid_results: Count the number of requests that satisfy the given FTL and TPOT.
    """
    count = 0
    for req in request_results:
        if req.ftl <= ftl and req.tpot <= tpot:
            count += 1
    return count

def get_slo_attainment(request_results: list[RequestResult], ftl: float, tpot: float) -> float:
    """
    get_slo_attainment: Get the SLO attainment of the given request results under the given FTL and TPOT.
    """
    return count_valid_results(request_results, ftl, tpot) / len(request_results)
