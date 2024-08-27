# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Input from soundcard hardware using the SoundDevice library.

.. autosummary::
    :toctree: generated/

    SoundDeviceSamplesGenerator
"""

from traits.api import Any, Bool, Float, Int, Long, Property, Trait, cached_property, observe

from .configuration import config
from .internal import digest
from .tprocess import SamplesGenerator

if config.have_sounddevice:
    import sounddevice as sd


class SoundDeviceSamplesGenerator(SamplesGenerator):
    """Controller for sound card hardware using sounddevice library.

    Uses the device with index :attr:`device` to read samples
    from input stream, generates output stream via the generator
    :meth:`result`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if config.have_sounddevice is False:
            msg = 'SoundDevice library not found but is required for using the SoundDeviceSamplesGenerator class.'
            raise ImportError(msg)

    #: input device index, refers to sounddevice list
    device = Int(0, desc='input device index')

    #: Number of input channels, maximum depends on device
    numchannels = Long(1, desc='number of analog input channels that collects data')

    #: Number of samples to collect; defaults to -1.
    # If is set to -1 device collects till user breaks streaming by setting Trait: collectsamples = False
    numsamples = Long(-1, desc='number of samples to collect')

    #: Indicates if samples are collected, helper trait to break result loop
    collectsamples = Bool(True, desc='Indicates if samples are collected')

    #: Sampling frequency of the signal, changes with sinusdevices
    sample_freq = Property(desc='sampling frequency')

    _sample_freq = Float(default_value=None)

    #: Datatype (resolution) of the signal, used as `dtype` in a sd `Stream` object
    precision = Trait('float32', 'float16', 'int32', 'int16', 'int8', 'uint8', desc='precision (resolution) of signal')

    #: Indicates that the sounddevice buffer has overflown
    overflow = Bool(False, desc='Indicates if sounddevice buffer overflow')

    #: Indicates that the stream is collecting samples
    running = Bool(False, desc='Indicates that the stream is collecting samples')

    #: The sounddevice InputStream object for inspection
    stream = Any

    # internal identifier
    digest = Property(depends_on=['device', 'numchannels', 'numsamples'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    # checks that numchannels are not more than device can provide
    @observe('device,numchannels')
    def _get_numchannels(self, event):  # noqa ARG002
        self.numchannels = min(self.numchannels, sd.query_devices(self.device)['max_input_channels'])

    def _get_sample_freq(self):
        if self.sample_freq is not None:
            return self._sample_freq
        return sd.query_devices(self.device)['default_samplerate']

    def _set_sample_freq(self, f):
        self._sample_freq = f

    def device_properties(self):
        """Returns
        -------
        Dictionary of device properties according to sounddevice
        """
        return sd.query_devices(self.device)

    def result(self, num):
        """Python generator that yields the output block-wise. Use at least a
        block-size of one ring cache block.

        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).

        Returns
        -------
        Samples in blocks of shape (num, :attr:`numchannels`).
            The last block may be shorter than num.

        """
        print(self.device_properties(), self.sample_freq)
        self.stream = stream_obj = sd.InputStream(
            device=self.device,
            channels=self.numchannels,
            clip_off=True,
            samplerate=self.sample_freq,
            dtype=self.precision,
        )

        with stream_obj as stream:
            self.running = True
            if self.numsamples == -1:
                while self.collectsamples:  # yield data as long as collectsamples is True
                    data, self.overflow = stream.read(num)
                    yield data[:num]

            elif self.numsamples > 0:  # amount of samples to collect is specified by user
                samples_count = 0  # numsamples counter
                while samples_count < self.numsamples:
                    anz = min(num, self.numsamples - samples_count)
                    data, self.overflow = stream.read(num)
                    yield data[:anz]
                    samples_count += anz
        self.running = False
        return
