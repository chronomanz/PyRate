'''
Contains objects common to multiple parts of PyRate  

Created on 12/09/2012
@author: bpd900
'''

import os
import gdal, gdalconst

import roipac

# TODO: add phase_data and amplitude_data properties?
#     Problem: property links to FULL dataset, which may be slower than row by row access
#         row by row access would be efficient, but needes wavelength converted layers


class Ifg(object):
	"""Interferogram class, representing the difference between two acquisitions.
	Ifg objects double as a container for related data."""

	def __init__(self, path, hdr_path=None):
		if hdr_path:
			# handle non default header (eg. for look files in other formats)
			self.data_path, self.hdr_path = path, hdr_path
		else:
			# default the header path
			self.data_path, self.hdr_path = roipac.filename_pair(path)
		
		header = roipac.parse_header(self.hdr_path)
		self.ehdr_path = None # path to EHdr format header

		# dynamically include header items as class attributes
		for key, value in header.iteritems():
			if self.__dict__.has_key(key):
				msg = "Attribute %s already exists for %s" % (key, path)
				raise Exception(msg)
			self.__dict__[key] = value

		self.dataset = None # for GDAL dataset obj
		self._amp_band = None
		self._phase_band = None

		# TODO: what are these for?
		self.max_variance = None
		self.alpha = None
		self.nodata_fraction = None


	def __str__(self):
		return "Ifg('%s')" % self.data_path


	def __repr__(self):
		return "Ifg('%s', '%s')" % (self.data_path, self.hdr_path)


	def open(self, readonly=True):
		'''Opens a interferogram dataset for reading. Creates ESRI/EHdr format
		header in the data dir, so GDAL has access to recognised header.'''
		if self.ehdr_path is None:
			self.ehdr_path = roipac.to_ehdr_header(self.hdr_path)			
			if readonly:
				self.dataset = gdal.Open(self.data_path)
			else:
				self.dataset = gdal.Open(self.data_path, gdalconst.GA_Update)
			
			if self.dataset is None:
				raise IfgException("Error opening %s" % self.data_path)
			
		else:
			if self.dataset is not None:
				msg = "open() already called for %s" % self
				raise IfgException(msg)


	@property
	def amp_band(self):
		if self._amp_band is not None:
			return self._amp_band
		else:
			if self.dataset is not None:
				self._amp_band = self.dataset.GetRasterBand(1)
				return self._amp_band
			else:
				raise IfgException("Ifg %s has not been opened" % self.data_path)


	@property
	def phase_band(self):
		if self._phase_band is not None:
			return self._phase_band
		else:
			if self.dataset is not None:
				self._phase_band = self.dataset.GetRasterBand(2)
				return self._phase_band
			else:
				raise IfgException("Ifg %s has not been opened" % self.data_path)


class IfgException(Exception):
	'''Generic exception class for interferogram errors'''
	pass

class PyRateException(Exception):
	'''Generic exception class for PyRate S/W errors'''
	pass

