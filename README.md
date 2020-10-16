# gridspace-stanford-harper-valley
The Gridspace-Stanford Harper Valley speech dataset. Created in support of CS224S.

Directory Structure:

```
data
    audio
        agent
            <sid1>.wav
            <sid2>.wav
            ...
        caller
            <sid1>.wav
            <sid2>.wav
            ...
    metadata
        <sid1>.json
        <sid2>.json
        ...
    transcript
        <sid1>.json
        <sid2>.json
        ...
```

Each conversation has an id referred to as it's sid.  All associated files are named based on that sid.
Each conversation has four associated files, two audio files, one transcript file and one metadata file.
The audio for each conversation is divided in to two single channel wav files, available under the audio/agent and audio/caller directories.

The transcript json files are lists of segment where each segment is a json object with the following schema:

```json
{
    "channel_index": 2,
    "dialog_acts": [
        "gridspace_greeting"
    ],
    "duration_ms": 2280,
    "emotion": {
        "neutral": 0.33766093850135803,
        "negative": 0.024230705574154854,
        "positive": 0.6381083130836487
    },
    "human_transcript": "hello this is harper valley national bank",
    "index": 1,
    "offset_ms": 5990,
    "speaker_role": "agent",
    "start_ms": 3990,
    "start_timestamp_ms": 1591056064136,
    "transcript": "hello this is harper valley national bank",
    "word_durations_ms": [
        330,
        150,
        120,
        330,
        270,
        420,
        330
    ],
    "word_offsets_ms": [
        0,
        660,
        810,
        930,
        1260,
        1530,
        1950
    ]
}
```

channel_index: This identifies the channel (1 is caller side and 2 is agent)
dialog_acts: This is a list of tags assigned by Gridspace's Dialog Act model.  
Possible tags are
    - gridspace_greeting
    - ...

duration_ms: Duration of the call in milliseconds.
emotion: Softmax output of Gridspace's Emotion model, determining whether the emotional valence of the segment was positive, negative, or neutral.
human_transcript: Corrected transcript as determined by transcriptionists.
index: Index of the segment within the conversation.
offset_ms: Offset of the start of the segment from the beginning of the recording
speaker_role: Whether the speaker is the agent or the caller
start_ms: Offset of the start of the segment from the beginning of the conversation
start_timestamp_ms: Start of the segment in milliseconds since the epoch
transcript: Machine generated transcript
word_duration_ms: List of durations of the words in the machine generated transcript
word_offsets_ms: List of offsets of words from the segment's start in the machine generated transcript

Each metadata file has the following schema:
```json
{
    "agent": {
        "arrival_time_ms": 1591056052086,
        "hangup_time_ms": 1591056113219,
        "metadata": {
            "agent_name": "Linda"
        },
        "responses": [
            {
                "submit_time_ms": 1591056109505,
                "data": {
                    "task_type": "get branch hours"
                }
            }
        ],
        "speaker_id": 40,
        "survey_response": {
            "submit_time_ms": 1591056115557,
            "data": {}
        }
    },
    "caller": {
        "arrival_time_ms": 1591056040570,
        "hangup_time_ms": 1591056112224,
        "metadata": {
            "first and last name": "Robert Miller"
        },
        "responses": [
            {
                "submit_time_ms": 1591056087001
            }
        ],
        "speaker_id": 60,
        "survey_response": {
            "submit_time_ms": 1591056115396,
            "data": {
                "ease_of_connection": "10",
                "partner_rating": "10"
            }
        }
    },
    "end_time_ms": 1591056112109,
    "sid": "ff0296d00e5e4184",
    "start_time_ms": 1591056058046,
    "labels": {
        "lhvb_script": 5.0,
        "caller_mos": 5.0,
        "agent_mos": 5.0
    },
    "session": "Little Harper Valley 3",
    "tasks": [
        {
            "branch hours": "8:30 am to 5:00 pm",
            "task_type": "get branch hours"
        }
    ]
}
```

agent:
    arrival_time_ms: Unix timestamp for when the agent called in to Gridspace Mixer
    hangup_time_ms: Unix timestamp for when the agent hung up
    metadata: Metadata provided to the agent by Gridspace Mixer during the conversation
    responses: List of responses submitted by the agent during the conversation each with the following keys:
        submit_time_ms: Unix timestamp for when the data was submitted
        data: Data submitted by the agent. Will include task_type and possibly other details 
    speaker_id: Speaker ID for the agent
    survey_response: 
        submit_time_ms: Unix timestamp for when the agent submitted the survey
        data: Data submitted during the agent's survey. Will be blank if no survey was submitted
caller:
    arrival_time_ms: Unix timestamp for when the caller called in to Gridspace Mixer
    hangup_time_ms: Unix timestamp for when the caller hung up
    metadata: Metadata provided to the caller by Gridspace Mixer during the conversation
    responses: List of responses submitted by the caller during the conversation.  This indicates when the Caller thought the agent had completed their task
    speaker_id: Speaker ID for the caller
    survey_response: 
        submit_time_ms: Unix timestamp for when the caller submitted the survey
        data: Data submitted during the caller's survey. Will be blank if no survey was submitted
end_time_ms: Unix timestamp for when the conversation ended
sid: Identifier for the conversation
start_time_ms: Unix timestamp for when the conversation started
labels: Labels assigned by the transcriptionists for the following categories
    lhvb_script: How well the caller's stuck to the provided script
    caller_mos: How well could the caller be understood
    agent_mos: How well could the agent be understood
session: Which session this conversation was a part of.  There were three Sessions in total.
tasks: Indicates which task was assigned to the caller.
    

Here is a list of the possible tasks that were assigned:
    - get branch hours
    - ...
    
